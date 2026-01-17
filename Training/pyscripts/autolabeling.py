from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
from autodistill_yolov8 import YOLOv8
from autodistill.utils import plot
import cv2

def main():
    base_model = GroundingDINO(ontology=CaptionOntology({"yellow ball": "ball"}), box_threshold=0.2, text_threshold=0.95)
    base_model.label(
        input_folder="./datasets/real_gamepieces",
        extension=".jpg",
        output_folder="./datasets/autolabeled-rebuilt-self-collected"
    )

if __name__ == "__main__":
    main()
