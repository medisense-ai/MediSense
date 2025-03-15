import torch
from models import get_model

# Suppose you used "mammo-segmentation-unet" with n_channels=1, n_classes=7
model_class = get_model("mammo-segmentation-unet", categories=1)
model = model_class(n_channels=1, n_classes=7)
model.load_state_dict(torch.load("best_mammo_segmentation_unet.pth"))
print(model.eval())

# Now you can do inference on the validation set to inspect predicted masks
# or you can do a quick loop to measure metrics again, etc.
