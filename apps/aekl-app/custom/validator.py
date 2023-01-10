from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from flip import FLIP
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.metrics import compute_meandice
from monai import transforms
from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants

from simple_network import SimpleNetwork


class FLIP_VALIDATOR(Executor):
    def __init__(self, validate_task_name=AppConstants.TASK_VALIDATION, project_id="", query=""):
        super(FLIP_VALIDATOR, self).__init__()

        self._validate_task_name = validate_task_name

        # Setup the model
        self.model = SimpleNetwork()
        self.device = torch.device("cuda")
        self.model.to(self.device)

        self.val_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image"], reader="NiBabelReader", as_closest_canonical=False),
                transforms.AddChanneld(keys=["image"]),
                transforms.ScaleIntensityRanged(keys=["image"], a_min=-15, a_max=100, b_min=0, b_max=1, clip=True),
                transforms.ToTensord(keys=["image"]),
            ]
        )

        # Setup the training dataset
        self.flip = FLIP()
        self.project_id = project_id
        self.query = query
        self.dataframe = self.flip.get_dataframe(self.project_id, self.query)

    def get_datalist(self, dataframe, val_split=0.2):
        """Returns datalist for validation."""
        _, val_dataframe = np.split(dataframe, [int((1 - val_split) * len(dataframe))])

        datalist = []
        for accession_id in val_dataframe["accession_id"]:
            try:
                accession_folder_path = self.flip.get_by_accession_number(self.project_id, accession_id)

                all_images = list(Path(accession_folder_path).rglob("*.nii*"))
                for image in all_images:
                    header = nib.load(str(image))

                    # check is 3D and at least 128x128x128 in size
                    if len(header.shape) == 3 and all([dim >= 128 for dim in header.shape]):
                        datalist.append({"image": str(image)})
            except:
                pass
        print(f"Found {len(datalist)} files in val")
        return datalist

    def local_validation(self, weights, abort_signal):
        self.model.load_state_dict(weights)

        self.model.eval()
        total_mean_dice = 0
        num_images = 0
        with torch.no_grad():
            for i, batch in enumerate(self._test_loader):
                if abort_signal.triggered:
                    # If abort_signal is triggered, we simply return.
                    # The outside function will check it again and decide steps to take.
                    return

                images = batch["image"].to(self.device)

                output_logits = sliding_window_inference(
                    images,
                    sw_batch_size=2,
                    roi_size=(128, 128, 128),
                    predictor=self.model,
                    overlap=0.25,
                    do_sigmoid=False,
                )
                output = torch.sigmoid(output_logits)
                metric = compute_meandice(output, include_background=False).cpu().numpy()

                total_mean_dice += metric.sum()
                num_images += images.size()[0]
                print(f"Validator Iteration: {i}, Metric: {total_mean_dice}, Num Images: {num_images}")

            metric = total_mean_dice / float(num_images)

        return metric

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:

        test_dict = self.get_image_and_label_list(self.dataframe)
        self._test_dataset = Dataset(test_dict, transform=self.val_transforms)
        self._test_loader = DataLoader(self._test_dataset, batch_size=1, shuffle=False)

        if task_name == self._validate_task_name:
            model_owner = "?"
            try:
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(fl_ctx, "Error in extracting dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Ensure data_kind is weights.
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_exception(
                        fl_ctx,
                        f"DXO is of type {dxo.data_kind} but expected type WEIGHTS.",
                    )
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Extract weights and ensure they are tensor.
                model_owner = shareable.get_header(AppConstants.MODEL_OWNER, "?")
                weights = {k: torch.as_tensor(v, device=self.device) for k, v in dxo.data.items()}

                # Get validation accuracy
                val_accuracy = self.local_validation(weights, abort_signal)
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                self.log_info(
                    fl_ctx,
                    f"Accuracy when validating {model_owner}'s model on"
                    f" {fl_ctx.get_identity_name()}"
                    f"s data: {val_accuracy}",
                )

                dxo = DXO(data_kind=DataKind.METRICS, data={"val_acc": val_accuracy})
                return dxo.to_shareable()
            except:
                self.log_exception(fl_ctx, f"Exception in validating model from {model_owner}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)
