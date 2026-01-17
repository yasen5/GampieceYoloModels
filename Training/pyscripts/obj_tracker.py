from model_tester import ONNXDetector
import constants
import cv2
from dataclasses import dataclass
from typing import Deque
from collections import deque

@dataclass(frozen=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    cls: int

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[int, int]:
        return (
            (self.x1 + self.x2) // 2,
            (self.y1 + self.y2) // 2,
        )

@dataclass
class Track:
    history: Deque[BoundingBox]
    missed_frames: int = 0

    @property
    def last(self) -> BoundingBox:
        return self.history[-1]

    @property
    def cls(self) -> int:
        return self.last.cls

def compute_iou(a: BoundingBox, b: BoundingBox) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    union_area = a.area + b.area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

def parse_detections(raw) -> list[BoundingBox]:
    boxes = []
    for x1, y1, x2, y2, conf, cls in raw:
        if x1 == 0 and x2 == 0:
            continue
        boxes.append(BoundingBox(x1, y1, x2, y2, cls))
    return boxes

def run_tracking():
    MIN_IOU = 0.1;

    detector = ONNXDetector(constants.MODEL_PATH)

    cap = cv2.VideoCapture("datasets/videos/480pvid.mp4")

    object_position_lists: list[Track] = [] 

    counter = 0;

    while (True):
        counter += 1;
        ok, frame = cap.read();
        if not ok:
            print("Done")
            break;
        print(f"Frame {counter}\tObjects: {len(object_position_lists)}")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB);

        raw_detections = detector.detect(rgb);
        detections = parse_detections(raw_detections);

        claimed = [False] * len(object_position_lists);
        num_possible_objects = len(object_position_lists);

        for tracked in object_position_lists:
            tracked.missed_frames += 1;

        for box in detections:
            if box.area == 0:
                continue;
            
            closest_obj_index = -1;
            closest_iou = 0;
            for i in range(num_possible_objects):
                if claimed[i] or object_position_lists[i].cls != box.cls:
                    continue;
                iou = compute_iou(box, object_position_lists[i].last);
                if (iou > MIN_IOU and iou > closest_iou):
                    closest_iou = iou;
                    closest_obj_index = i;
            if closest_obj_index != -1:
                object_position_lists[closest_obj_index].history.append(box);
                object_position_lists[closest_obj_index].missed_frames = 0;
                claimed[closest_obj_index] = True;
            else:
                object_position_lists.append(Track(deque(maxlen=15), 0));
                object_position_lists[-1].history.append(box);
                claimed.append(True);
        detector.draw_detections(frame, raw_detections);
        object_position_lists = [x for x in object_position_lists if x.missed_frames <= 3];
        for object_pos_list in object_position_lists:
            if len(object_pos_list.history) < 2:
                continue;
            for i in range(1, len(object_pos_list.history)):
                new_center = object_pos_list.history[i].center;
                old_center = object_pos_list.history[i-1].center;
                cv2.line(frame, old_center, new_center, (0, 255, 0), 16);
        cv2.imshow("Test", frame);
        cv2.waitKey(0);
    cv2.destroyAllWindows(); 
            
if __name__=="__main__":
    run_tracking();
