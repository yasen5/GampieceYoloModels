from ultralytics import YOLO

def main():
    # Path to your trained model
    best_model_path = "runs/detect/train2/weights/best.pt"

    # Load it
    model = YOLO(best_model_path)

    # Export to ONNX
    export_path = model.export(
        format="onnx",
        simplify=True,     # Simplify the exported graph
        dynamic=False,     # Set to True if you want variable image sizes
        imgsz=640,         # Match your training size
        device=0,          # GPU if available
        nms=True           # Include NMS if you want postprocessing in ONNX
    )

    print(f"ONNX model exported to: {export_path}")

if __name__ == "__main__":
    main()
