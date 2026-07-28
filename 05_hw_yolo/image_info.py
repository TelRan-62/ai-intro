import os
from ultralytics import YOLO
from pandas import DataFrame
import math

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

class ImageInfo :
    def __init__(self, path_image):
        model = YOLO(os.path.join(_MODULE_DIR, 'yolov8m-seg.pt'))
        self._all_names = model.names
        if not os.path.isabs(path_image):
            path_image = os.path.join(_MODULE_DIR, path_image)
        self._boxes = model(path_image, device="cpu")[0].boxes
        self._class_indices: dict[str, list[int]] = self._get_class_indices_dict()
    def  _get_class_indices_dict(self):
        res: dict[str, list[int]] = {}
        [self._update_class_indices(bi, box, res) for bi, box in enumerate(self._boxes)]
        return res 
    def _update_class_indices(self, bi: int, box, res: dict[str, list[int]]):
        class_name: str = self._all_names[box.cls.item()]
        res.setdefault(class_name, []).append(bi)
    def _get_data_frame(self)-> DataFrame:
        xyxy = self._boxes.xyxy.cpu().numpy()
        cls = self._boxes.cls.cpu().numpy()
        conf = self._boxes.conf.cpu().numpy()
        names = [self._all_names[c] for c in cls]
        df = DataFrame(xyxy, columns=["xmin", "ymin", "xmax", "ymax"])
        df["name"] = names
        df["confidence"] = conf 
        return df
    def _get_distance_between(self, box_ind1: int, box_ind2: int)->float :
        x1_center, y1_center, *_ = self._boxes[box_ind1].xywhn[0].tolist()
        x2_center, y2_center, *_ = self._boxes[box_ind2].xywhn[0].tolist()
        return math.hypot(x1_center - x2_center, y1_center - y2_center)
    def _get_min_distance(self, box_index: int, box_indices: list[int]) -> tuple[int, float]:
        res: tuple[int, float] = (box_indices[0], self._get_distance_between(box_index, box_indices[0]))
        for bi in box_indices:
            if (d := self._get_distance_between(box_index, bi)) < res[1]:
                res = (bi, d)
        return res        
    def _get_belongings_person_dict(self):
        belonging_indices = self.boxes_class("suitcase") + self.boxes_class("handbag")
        person_indices = self.boxes_class("person")
        res: dict[int, tuple[int, float]] = {bi: self._get_min_distance(bi, person_indices)
                                             for bi in belonging_indices}
        return res
    def boxes_class(self, class_name: str) -> list[int]:
        return self._class_indices.get(class_name, [])
    def box_info(self, index: int)-> tuple[float, float, float, float, float, str]:
        box = self._boxes[index]
        res: list = box.xyxy[0].tolist()
        res.append(box.conf.item())
        res.append(self._all_names[box.cls.item()])
        return tuple(res)
    def data_frame(self):
        if not hasattr(self, "_df"):
            self._df = self._get_data_frame()
        return self._df
    def suitcase_handbag_person(self, threshold: float) -> dict[int, tuple[int, float] | None]:
        if not hasattr(self, "_belonging_person"):
            self._belonging_person: dict[int, tuple[int, float]] = self._get_belongings_person_dict()
        belong_person: dict[int, tuple[int, float]] = \
        {bi: (pi,d) for bi, (pi, d) in self._belonging_person.items()
         if d <= threshold} 
        no_belong_person:dict[int, None] = \
            {bi: None for bi, (_, d) in self._belonging_person.items()
             if d > threshold}
        return belong_person | no_belong_person
            
               
        