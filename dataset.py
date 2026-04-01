import json
from monai.data import DataLoader, Dataset, CacheDataset

def load_datalists(train_json, val_json):
    with open(train_json) as f:
        train_files = json.load(f)["training"]
    with open(val_json) as f:
        val_files = json.load(f)["training"]
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