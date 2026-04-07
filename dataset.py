
from monai.data.dataset import CacheDataset
from monai.data.dataloader import DataLoader

import os
from sklearn.model_selection import train_test_split
""" load_datalists(train_json, val_json):
    with open(train_json) as f:
        train_files = json.load(f)["training"]
    with open(val_json) as f:
        val_files = json.load(f)["training"]
    return train_files, val_files
"""

def load_datalists(data_dir, val_fraction=0.2, seed=42):
    cases = sorted(
        [d for d in os.scandir(data_dir) if d.is_dir()],
        key=lambda d: d.name
    )

    def make_entry(case):
        n = case.name
        return {
            "image": [
                os.path.join(case.path, f"{n}-t1c.nii.gz"),
                os.path.join(case.path, f"{n}-t1n.nii.gz"),
                os.path.join(case.path, f"{n}-t2f.nii.gz"),
                os.path.join(case.path, f"{n}-t2w.nii.gz"),
            ],
            "label": os.path.join(case.path, f"{n}-seg.nii.gz"),
        }

    all_files = [make_entry(c) for c in cases]
    train_files, val_files = train_test_split(
        all_files, test_size=val_fraction, random_state=seed
    )
    return train_files, val_files

def get_dataloaders(train_files, val_files, train_transforms, val_transforms,
                    batch_size, cache_rate=0.1, num_workers=4):
    train_ds = CacheDataset(train_files, transform=train_transforms,
                            cache_rate=cache_rate, num_workers=num_workers)
    val_ds   = CacheDataset(val_files,   transform=val_transforms,
                            cache_rate=cache_rate, num_workers=num_workers)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=1,          shuffle=False, num_workers=num_workers)
    return train_loader, val_loader