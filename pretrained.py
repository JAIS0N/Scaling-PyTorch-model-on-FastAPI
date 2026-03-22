import torch
import torchvision.models as models

model_name = "mobilenet_v2.pt"

# Load a pre-trained MobileNetV2 model
model = models.mobilenet_v2(pretrained=True)
# Save the model
torch.save(model, model_name)

print(f"Model exported successfully as {model_name}")
