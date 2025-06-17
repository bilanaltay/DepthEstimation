import torch

from PIL import Image
from misc import colorize

class DepthEstimationModel:
    def __init__(self) -> None:
        self.device = self._get_device()
        self.model = self._init_model("isl-org/ZoeDepth","ZoeD_N").to(self.device)

    def _get_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _init_model(self, model_repo="intel-org/ZoeDepth", model_name="ZoeD_N"):
        torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo
        model = torch.hub.load(model_repo, model_name, pretrained=True, skip_validation=False)
        model.eval()
        print("model initialized")
        return model

    def save_colored_depth(self, depht_numpy, output_path):
        colored = colorize(depht_numpy)
        Image.fromarray(colored).save(output_path)
        print("Saved colored depth to", output_path)

    def calculate_depthmap(self, img_path, output_path):
        image = Image.open(img_path).convert("RGB")
        depth_numpy = self.model.infer_pil(image)
        self.save_colored_depth(depth_numpy, output_path)
        return f"image saved {output_path}.png"

model = DepthEstimationModel()
model.calculate_depthmap("./sea.png","output_sea.png")