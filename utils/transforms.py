from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.utility.dictionary import EnsureChannelFirstd
from monai.transforms.intensity.dictionary import NormalizeIntensityd
from monai.transforms.spatial.dictionary import RandFlipd, RandRotate90d
from monai.transforms.croppad.dictionary import RandSpatialCropd
from monai.transforms.utility.dictionary import ToTensord
import numpy as np

class RandomModalityMaskd:
    """Zero out 0-MAX_MASKED modality channels with probability MASK_PROB."""

    def __init__(self, keys, mask_prob=0.5, max_masked=2):
        self.keys      = keys
        self.mask_prob = mask_prob
        self.max_masked = max_masked

    def __call__(self, data):
        if np.random.rand() < self.mask_prob:
            n_mask = np.random.randint(1, self.max_masked + 1)
            channels = np.random.choice(len(self.keys), n_mask, replace=False)
            for key in self.keys:
                img = data[key]
                for c in channels:
                    img[c] = 0.0          # zero-out selected channel
                data[key] = img
        return data


def get_train_transforms(roi_size, mask_prob, max_masked):
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandSpatialCropd(keys=["image", "label"], roi_size=roi_size, random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandomModalityMaskd(keys=["image"], mask_prob=mask_prob, max_masked=max_masked),
        ToTensord(keys=["image", "label"]),
    ])


def get_val_transforms(roi_size):
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
    ])