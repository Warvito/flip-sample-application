from pathlib import Path
import json

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms
from monai.data import DataLoader, Dataset
from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model import make_model_learnable, model_learnable_to_dxo
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.pt.pt_fed_utils import PTModelPersistenceFormatManager
from torch.cuda.amp import GradScaler, autocast

from flip import FLIP
from perceptual_loss import PerceptualLoss
from pt_constants import PTConstants
from simple_network import kl_loss_function, PatchAdversarialLoss, SimpleNetwork


class FLIP_TRAINER(Executor):
    def __init__(
        self,
        lr=1,
        epochs=1,
        train_task_name=AppConstants.TASK_TRAIN,
        submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL,
        exclude_vars=None,
        project_id="",
        query="",
    ):
        super().__init__()
        self._train_task_name = train_task_name
        self._submit_model_task_name = submit_model_task_name
        self._exclude_vars = exclude_vars

        self.config = {}
        working_dir = Path(__file__).parent.resolve()
        with open(str(working_dir / "config.json")) as file:
            self.config = json.load(file)

        self.model = SimpleNetwork()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer_g = torch.optim.Adam(params=self.model.autoencoder.parameters(), lr=1e-5)
        self.optimizer_d = torch.optim.Adam(params=self.model.discriminator.parameters(), lr=5e-5)

        self.scaler_g = GradScaler()
        self.scaler_d = GradScaler()

        self.kl_weight = 1e-6
        self.perceptual_weight = 0.001
        self.adv_weight = 0.01

        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")
        self.perceptual_loss = PerceptualLoss(
            spatial_dims=3, network_type="squeeze", fake_3d_ratio=0.3, is_fake_3d=True
        )
        self.perceptual_loss.to(self.device)

        # Setup transforms using dictionary-based transforms.
        self._train_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image"], reader="NiBabelReader", as_closest_canonical=False),
                transforms.EnsureChannelFirstd(keys=["image"]),
                transforms.ScaleIntensityRanged(keys=["image"], a_min=-15, a_max=100, b_min=0, b_max=1, clip=True),
                transforms.CenterSpatialCropd(keys=["image"], roi_size=(160, 160, 80)),
                transforms.Resized(keys=["image"], spatial_size=(80, 80, 40)),
                transforms.ToTensord(keys=["image"]),
            ]
        )

        # Setup the training dataset
        self.flip = FLIP()
        self.project_id = project_id
        self.query = query
        self.dataframe = self.flip.get_dataframe(self.project_id, self.query)

        # Setup the persistence manager to save PT model.
        # The default training configuration is used by persistence manager in case no initial model is found.
        self._default_train_conf = {"train": {"model": type(self.model).__name__}}
        self.persistence_manager = PTModelPersistenceFormatManager(
            data=self.model.state_dict(), default_train_conf=self._default_train_conf
        )

        self.project_id = project_id
        self.query = query

    def get_datalist(self, dataframe, val_split=0.2):
        train_dataframe, _ = np.split(dataframe, [int((1 - val_split) * len(dataframe))])

        datalist = []
        for accession_id in train_dataframe["accession_id"]:
            image_data_folder_path = self.flip.get_by_accession_number(self.project_id, accession_id)
            accession_folder_path = Path(image_data_folder_path) / accession_id

            for image in list(accession_folder_path.rglob("*.nii*")):
                header = nib.load(str(image))

                # check is 3D and at least 128x128x128 in size
                if len(header.shape) == 3 and all([dim >= 128 for dim in header.shape]):
                    datalist.append({"image": str(image)})

        print(f"Found {len(datalist)} files in the training set")
        return datalist

    def local_train(self, fl_ctx, weights, abort_signal):
        self.model.load_state_dict(state_dict=weights)
        self.model.train()

        for epoch in range(self.config["LOCAL_ROUNDS"]):
            epoch_recons_loss = 0
            epoch_discriminator_loss = 0
            for step, batch in enumerate(self._train_loader):

                if abort_signal.triggered:
                    # If abort_signal is triggered, we simply return.
                    # The outside function will check it again and decide steps to take.
                    return

                images = batch["image"].to(self.device)

                # Generator part
                self.optimizer_g.zero_grad(set_to_none=True)
                with autocast(enabled=True):
                    reconstruction, z_mu, z_sigma = self.model.autoencoder(x=images)
                    l1_loss = F.l1_loss(reconstruction.float(), images.float())
                    kl_loss = kl_loss_function(z_mu, z_sigma)

                    p_loss = self.perceptual_loss(reconstruction.float(), images.float())

                    logits_fake = self.model.discriminator(reconstruction.contiguous().float())[-1]
                    generator_loss = self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)

                    loss = (
                        l1_loss
                        + self.kl_weight * kl_loss
                        + self.perceptual_weight * p_loss
                        + self.adv_weight * generator_loss
                    )

                self.scaler_g.scale(loss).backward()
                self.scaler_g.unscale_(self.optimizer_g)
                torch.nn.utils.clip_grad_norm_(self.model.autoencoder.parameters(), 1)
                self.scaler_g.step(self.optimizer_g)
                self.scaler_g.update()

                # Discriminator part
                self.optimizer_d.zero_grad(set_to_none=True)

                with autocast(enabled=True):
                    logits_fake = self.model.discriminator(reconstruction.contiguous().detach())[-1]
                    loss_d_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                    logits_real = self.model.discriminator(images.contiguous().detach())[-1]
                    loss_d_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                    discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                    d_loss = self.adv_weight * discriminator_loss
                    d_loss = d_loss.mean()

                self.scaler_d.scale(d_loss).backward()
                self.scaler_d.unscale_(self.optimizer_d)
                torch.nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), 1)
                self.scaler_d.step(self.optimizer_d)
                self.scaler_d.update()

                epoch_recons_loss += l1_loss.item()
                epoch_discriminator_loss += d_loss.item()

            self.flip.send_metrics_value("Loss", epoch_recons_loss / (step + 1), fl_ctx)
            self.flip.send_metrics_value("Disc_Loss", epoch_discriminator_loss / (step + 1), fl_ctx)

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:

        if task_name == self._train_task_name:
            train_dict = self.get_datalist(self.dataframe)
            self._train_dataset = Dataset(train_dict, transform=self._train_transforms)
            self._train_loader = DataLoader(self._train_dataset, batch_size=16, shuffle=True, num_workers=4)
            self._n_iterations = len(self._train_loader)

            # Get model weights
            dxo = from_shareable(shareable)

            # Ensure data_kind is weights.
            if not dxo.data_kind == DataKind.WEIGHTS:
                return make_reply(ReturnCode.BAD_TASK_DATA)

            weights = {k: torch.as_tensor(v, device=self.device) for k, v in dxo.data.items()}
            self.local_train(fl_ctx, weights, abort_signal)

            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)

            self.save_local_model(fl_ctx)

            # Get the new state dict and send as weights
            new_weights = self.model.state_dict()
            new_weights = {k: v.cpu().numpy() for k, v in new_weights.items()}

            outgoing_dxo = DXO(
                data_kind=DataKind.WEIGHTS,
                data=new_weights,
                meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self._n_iterations},
            )
            return outgoing_dxo.to_shareable()

        elif task_name == self._submit_model_task_name:
            ml = self.load_local_model(fl_ctx)
            dxo = model_learnable_to_dxo(ml)
            return dxo.to_shareable()

        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)

    def save_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = Path(run_dir) / PTConstants.PTModelsDir
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / PTConstants.PTLocalModelName

        ml = make_model_learnable(self.model.state_dict(), {})
        self.persistence_manager.update(ml)
        torch.save(self.persistence_manager.to_persistence_dict(), str(model_path))

    def load_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = Path(run_dir) / PTConstants.PTModelsDir
        if not models_dir.exists():
            return None
        model_path = models_dir / PTConstants.PTLocalModelName

        self.persistence_manager = PTModelPersistenceFormatManager(
            data=torch.load(str(model_path)), default_train_conf=self._default_train_conf
        )
        ml = self.persistence_manager.to_model_learnable(exclude_vars=self._exclude_vars)
        return ml
