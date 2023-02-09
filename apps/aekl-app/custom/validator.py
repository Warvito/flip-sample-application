from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms
from monai.data import DataLoader, Dataset
from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from torch.cuda.amp import autocast

from flip import FLIP
from simple_network import SimpleNetwork


class FLIP_VALIDATOR(Executor):
    def __init__(self, validate_task_name=AppConstants.TASK_VALIDATION, project_id="", query=""):
        super().__init__()

        self._validate_task_name = validate_task_name

        self.model = SimpleNetwork()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.val_transforms = transforms.Compose(
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

    def get_datalist(self, dataframe, val_split=0.2):
        _, val_dataframe = np.split(dataframe, [int((1 - val_split) * len(dataframe))])

        datalist = []
        for accession_id in val_dataframe["accession_id"]:
            try:
                image_data_folder_path = self.flip.get_by_accession_number(self.project_id, accession_id)
            except Exception as err:
                print(f"Could not get image data folder path for {accession_id}:")
                print(f"{err=}")
                print(f"{type(err)=}")
                print(f"{err.args=}")
                continue

            accession_folder_path = Path(image_data_folder_path) / accession_id

            for image in accession_folder_path.glob("**/*_space-IXI549Space_desc-rigid_ct.nii"):
                if "_acq-Brain5mm" in image.stem:
                    continue

                datalist.append({"image": str(image)})

        print(f"Found {len(datalist)} files in the validation set")
        return datalist

    def local_validation(self, weights, abort_signal):
        self.model.load_state_dict(state_dict=weights)
        self.model.eval()

        epoch_recons_loss = 0
        with torch.no_grad():
            for step, batch in enumerate(self._test_loader):

                if abort_signal.triggered:
                    # If abort_signal is triggered, we simply return.
                    # The outside function will check it again and decide steps to take.
                    return

                images = batch["image"].to(self.device)

                with autocast(enabled=True):
                    reconstruction, z_mu, z_sigma = self.model.autoencoder(x=images)
                    l1_loss = F.l1_loss(reconstruction.float(), images.float())

                epoch_recons_loss += l1_loss.item()

        return epoch_recons_loss / (step + 1)

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        model_owner = "?"
        if task_name == self._validate_task_name:
            test_dict = self.get_datalist(self.dataframe)
            self._test_dataset = Dataset(test_dict, transform=self.val_transforms)
            self._test_loader = DataLoader(self._test_dataset, batch_size=16, shuffle=False, num_workers=4)

            # Get model weights
            dxo = from_shareable(shareable)

            # Ensure data_kind is weights.
            if not dxo.data_kind == DataKind.WEIGHTS:
                return make_reply(ReturnCode.BAD_TASK_DATA)

            # Extract weights and ensure they are tensor.
            model_owner = shareable.get_header(AppConstants.MODEL_OWNER, "?")
            weights = {k: torch.as_tensor(v, device=self.device) for k, v in dxo.data.items()}

            validation_loss = self.local_validation(weights, abort_signal)
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)

            dxo = DXO(data_kind=DataKind.METRICS, data={"validation_loss": validation_loss})
            return dxo.to_shareable()

        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)
