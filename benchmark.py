import torch
import time
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# Load model
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model.eval()

# Input
input_tensor = torch.randn(1, 3, 224, 224)

# -----------------------------
# 1. Normal inference
# -----------------------------
start = time.time()
for _ in range(50):
    output = model(input_tensor)
end = time.time()

print("Normal avg time:", (end - start) / 50)


# -----------------------------
# 2. inference_mode
# -----------------------------
start = time.time()
for _ in range(50):
    with torch.inference_mode():
        output = model(input_tensor)
end = time.time()

print("Inference_mode avg time:", (end - start) / 50)


# -----------------------------
# 3. JIT + inference_mode
# -----------------------------
example_input = torch.randn(1, 3, 224, 224)

jit_model = torch.jit.trace(model, example_input)
jit_model = torch.jit.freeze(jit_model)

start = time.time()
for _ in range(50):
    with torch.inference_mode():
        output = jit_model(input_tensor)
end = time.time()

print("JIT + inference_mode avg time:", (end - start) / 50)