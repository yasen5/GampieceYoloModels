import onnxruntime as ort
import cv2
import yaml
import numpy as np
import sys
from pathlib import Path

# Config
img_dir = Path("test_images")
MODEL_PATH = "best.onnx"

img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))

if len(img_files) == 0:
    print("ERROR: No images found!")
    exit(1)

session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
img_size = input_shape[2]

print(f"Input shape:{img_size}")

for img_file in img_files:
    # Load image
    img = cv2.imread(str(img_file))
    img_resized = cv2.resize(img, (img_size, img_size))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_input = np.transpose(img_norm, (2, 0, 1))[None, ...]

    # Inference
    outputs = session.run(None, {input_name: img_input})[0]
    num_preds = len(outputs)

    img_draw = img_resized.copy()

    print(f"Output shape: {outputs.shape}")
    for thingy in outputs:
        for det in thingy:
            print(f"Detection: {det}")
            x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow('Detections', img_draw)
        cv2.waitKey(0)
