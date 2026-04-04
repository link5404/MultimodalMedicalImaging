from pathlib import Path
import random
import numpy as np

from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.utility.dictionary import EnsureChannelFirstd
from monai.transforms.spatial.dictionary import Orientationd, Spacingd, RandFlipd, RandRotate90d
from monai.transforms.intensity.dictionary import ScaleIntensityRanged
from monai.transforms.croppad.dictionary import CropForegroundd, RandSpatialCropd
from monai.transforms.utility.dictionary import EnsureTyped
from monai.data.dataset import Dataset

MODALITIES = ["t1n", "t1c", "t2w", "t2f"]

class RandomModalityDropout:
    def __init__(self, keys, dropout_prob=0.5, max_drop_modalities=3):
        self.keys = keys
        self.dropout_prob = dropout_prob
        self.max_drop_modalities = max_drop_modalities
    def __call__(self, data):
        d = dict(data)
        if random.random() < self.dropout_prob:
            n_drop = random.randint(1, min(self.max_drop_modalities, len(self.keys)))
            drop_keys = random.sample(self.keys, n_drop)
            for k in drop_keys:
                d[k] = np.zeros_like(d[k])
        return d
    
def _find_one(patterns, case_dir):
    for p in patterns:
        files = list(case_dir.glob(p))
        if files:
            return str(files[0])
    raise FileNotFoundError(f"Missing file in {case_dir}: {patterns}")

def _build_items(data_root):
    data_root = Path(data_root)
    items = []
    
    # Find ALL subdirectories (no pattern restriction)
    case_dirs = [p for p in data_root.iterdir() if p.is_dir()]
    print(f"Found {len(case_dirs)} potential case folders")
    
    for case_dir in sorted(case_dirs):
        try:
            item = {}
            # Find modalities with multiple patterns
            modality_patterns = {
                't1n': ['*-t1n.nii.gz', '*-t1n.nii', '*t1n*.nii*', '*t1.nii*'],
                't1c': ['*-t1c.nii.gz', '*-t1c.nii', '*t1c*.nii*', '*t1ce*.nii*'],
                't2w': ['*-t2w.nii.gz', '*-t2w.nii', '*t2w*.nii*', '*t2*.nii*'],
                't2f': ['*-t2f.nii.gz', '*-t2f.nii', '*t2f*.nii*', '*flair*.nii*']
            }
            
            for m, patterns in modality_patterns.items():
                for pattern in patterns:
                    files = list(case_dir.glob(pattern))
                    if files:
                        item[m] = str(files[0])
                        break
            
            # Find ANY segmentation file
            seg_patterns = ['*seg*.nii*', '*label*.nii*', '*gt*.nii*']
            for pattern in seg_patterns:
                files = list(case_dir.glob(pattern))
                if files:
                    item['label'] = str(files[0])
                    break
            
            # Only add if we found all 4 modalities + label
            if all(m in item for m in MODALITIES) and 'label' in item:
                items.append(item)
            else:
                pass
                
        except Exception as e:
            print(f"X Error {case_dir.name}: {e}")
    
    print(f"Total valid cases: {len(items)}")
    return items

def build_datasets(data_root, image_size, val_split=0.2, seed=42, dropout_prob=0.5, max_drop_modalities=3):
    items = _build_items(data_root)
    rng = random.Random(seed)
    rng.shuffle(items)
    
    # TEMP: Use first 10 for train, skip val entirely
    train_items = items[:10]  
    val_items = items[10:12]  # Fixed indexing to avoid overlap
    
    train_transforms = Compose([
        LoadImaged(keys=MODALITIES + ['label']),
        EnsureChannelFirstd(keys=MODALITIES + ['label']),
        ScaleIntensityRanged(keys=MODALITIES, a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
        RandSpatialCropd(keys=MODALITIES + ['label'], roi_size=image_size, random_size=False),
        EnsureTyped(keys=MODALITIES + ['label']),
        RandomModalityDropout(keys=MODALITIES, dropout_prob=dropout_prob, max_drop_modalities=max_drop_modalities),
        EnsureTyped(keys=MODALITIES + ['label']),
    ])
    
    # Dummy val dataset (will be skipped)
    val_transforms = Compose([
        LoadImaged(keys=MODALITIES + ['label']),
        EnsureChannelFirstd(keys=MODALITIES + ['label']),
        ScaleIntensityRanged(keys=MODALITIES, a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
        RandSpatialCropd(keys=MODALITIES + ['label'], roi_size=image_size, random_size=False),
        EnsureTyped(keys=MODALITIES + ['label']),
        RandomModalityDropout(keys=MODALITIES, dropout_prob=dropout_prob, max_drop_modalities=max_drop_modalities),
        EnsureTyped(keys=MODALITIES + ['label']),
    ])
    print(f"DEBUG: train_items={len(train_items)}, val_items={len(val_items)}")
    return Dataset(train_items, train_transforms), Dataset(val_items, val_transforms)