
# Scaling PyTorch Model with FastAPI

This project demonstrates how to serve a PyTorch model using FastAPI and optimize it for high-performance inference on CPU, inspired by real-world production scenarios handling high request volumes.

---

## Overview

- Deploy a pretrained MobileNetV2 model
- Serve predictions via a FastAPI REST API
- Optimize inference using:
  - `torch.inference_mode()`
  - `model.eval()`
  - TorchScript JIT (Just-In-Time) compilation
- Containerize the application using Docker

---

## Key Features

- Fast inference with optimized CPU performance
- Lightweight deployment (no GPU required)
- Model loaded once at startup (efficient lifecycle)
- Benchmarking included (JIT vs non-JIT)
- Dockerized for portability

---

## Project Structure

```
project/
├── main.py
├── pretrained.py
├── benchmark.py
├── mobilenet_v2.pt
├── imagenet_class_index.json
├── Dockerfile
└── pyproject.toml / uv.lock (if using uv)
```

---

## Setup

### 1. Clone repository

```bash
git clone <your-repo>
cd <your-repo>
```

### 2. Install dependencies

Using uv:

```bash
uv sync
```

Or manually:

```bash
pip install fastapi uvicorn torch torchvision numpy pillow python-multipart
```

---

## Model Creation

```bash
uv run python pretrained.py
```

This will download the pretrained MobileNetV2 and save it as `mobilenet_v2.pt`.

---

## Run the API

```bash
uv run uvicorn main:app --reload
```

Open: `http://127.0.0.1:8000/docs`

---

## API Endpoints

**`GET /`** — Health check

**`GET /model`** — Returns model loading time

**`POST /predict`** — Upload an image and receive:
- Predicted class
- Confidence score
- Inference time

---

## Optimizations Used

### 1. Evaluation Mode

```python
model.eval()
```

Disables training layers like dropout.

### 2. Inference Mode

```python
with torch.inference_mode():
```

- Disables autograd completely
- Reduces memory usage
- Improves inference speed

### 3. TorchScript (JIT)

```python
model = torch.jit.trace(model, example_input)
model = torch.jit.freeze(model)
```

- Compiles model graph
- Reduces Python overhead
- Improves latency stability

---

## Benchmark Results

| Setup                 | Avg Time (seconds) |
|-----------------------|--------------------|
| Normal                | ~0.022             |
| inference_mode        | ~0.014             |
| JIT + inference_mode  | ~0.013             |

The major gain comes from `inference_mode`, with JIT providing additional optimization on top.

---

## Docker

### Build image

```bash
docker build -t pytorch-api .
```

### Run container

```bash
docker run -p 8000:80 pytorch-api
```

---

## Learnings

- Efficient model serving requires lifecycle management
- `inference_mode` provides significant performance improvement
- JIT improves consistency and reduces execution overhead
- CPU-based inference can be highly efficient with proper optimization

---

## Future Improvements

- Batch inference support
- Multi-worker scaling (Gunicorn + Uvicorn)
- ONNX / TensorRT optimization
- Deployment on cloud (AWS / Azure / GCP)

---

## Acknowledgements

Inspired by real-world production practices and optimization techniques in modern ML systems.

If you found this useful, give it a star and feel free to contribute.




<img width="671" height="152" alt="image" src="https://github.com/user-attachments/assets/e35b7032-d8d6-434b-86b3-89252f7fd3d1" />
