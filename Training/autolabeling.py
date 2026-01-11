from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
from autodistill_yolov8 import YOLOv8
from autodistill.utils import plot
import cv2

def main():
    base_model = GroundingDINO(ontology=CaptionOntology({"yellow ball": "ball"}), box_threshold=0.3, text_threshold=0.95)
    base_model.label(
        input_folder="./smaller-yellow",
        extension=".jpg",
        output_folder="./small-yellow-labeled"
    )

if __name__ == "__main__":
    main()
