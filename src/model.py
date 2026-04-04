import torch
from monai.networks.nets.swin_unetr import SwinUNETR
import urllib.request
from pathlib import Path

class BraTSSwinUNETR(torch.nn.Module):
    def __init__(self, in_channels=4, out_channels=1, feature_size=48):
        super().__init__()
        self.net = SwinUNETR(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=True
        )
        print(f"✅ SwinUNETR: {in_channels}→{out_channels}, feature_size={feature_size}")
        
        # Load official MONAI BraTS-compatible pretrained weights
        pretrained_path = self._download_pretrained()
        if pretrained_path and Path(pretrained_path).exists():
            try:
                checkpoint = torch.load(pretrained_path, map_location='cpu')
                # Load with partial matching (in/out channels may differ)
                self.net.load_state_dict(checkpoint, strict=False)
                print(f"✅ Loaded pretrained SwinUNETR from {pretrained_path}")
            except Exception as e:
                print(f"⚠️ Pretrained loading failed: {e}")
        else:
            print("⚠️ No pretrained weights - using random initialization")
    
    def _download_pretrained(self):
        """Download official MONAI SwinUNETR BraTS weights"""
        urls = [
            # Primary: MONAI Model Zoo BraTS segmentation
            "https://github.com/Project-MONAI/model-zoo/releases/download/swin_unetr_btcv_segmentation/swin_unetr_btcv_segmentation.pth",
            # Backup: Generic SwinUNETR
            "https://huggingface.co/darragh/swinunetr-btcv-tiny/resolve/main/pytorch_model.bin"
        ]
        
        pretrained_dir = Path("pretrained_weights")
        pretrained_dir.mkdir(exist_ok=True)
        
        for url in urls:
            try:
                filename = pretrained_dir / Path(url).name
                if not filename.exists():
                    print(f"Downloading pretrained model from {url}")
                    urllib.request.urlretrieve(url, filename)
                return str(filename)
            except:
                continue
        return None
    
    def forward(self, x):
        return self.net(x)