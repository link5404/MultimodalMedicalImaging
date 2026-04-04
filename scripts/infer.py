import torch
from monai.inferers.utils import sliding_window_inference
from MultimodalMedicalImaging.src.model import BraTSSwinUNETR

def predict(model_path, x, image_size=(128,128,128)):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BraTSSwinUNETR().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    x = x.to(device)
    with torch.no_grad():
        return sliding_window_inference(x, image_size, 1, model)
