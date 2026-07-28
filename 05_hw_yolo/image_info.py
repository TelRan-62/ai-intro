import math
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Callable, NamedTuple

from pandas import DataFrame
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

_MODULE_DIR = Path(__file__).resolve().parent
_DEFAULT_MODEL_NAME = "yolov8m-seg.pt"


class BoxInfo(NamedTuple):
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float
    name: str


@lru_cache(maxsize=None)
def _load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else _MODULE_DIR / path


def _distance(center1: tuple[float, float], center2: tuple[float, float]) -> float:
    return math.hypot(center1[0] - center2[0], center1[1] - center2[1])


def _min_distance_index(
    index: int,
    candidate_indices: list[int],
    center_of: Callable[[int], tuple[float, float]],
) -> tuple[int, float] | None:
    """Index (and distance) of the candidate closest to `index`, or None if there are no candidates."""
    if not candidate_indices:
        return None
    center = center_of(index)
    best_index = candidate_indices[0]
    best_distance = _distance(center, center_of(best_index))
    for candidate in candidate_indices[1:]:
        distance = _distance(center, center_of(candidate))
        if distance < best_distance:
            best_index, best_distance = candidate, distance
    return best_index, best_distance


def _apply_threshold(
    matches: dict[int, tuple[int, float] | None], threshold: float
) -> dict[int, tuple[int, float] | None]:
    return {
        bi: match if match is not None and match[1] <= threshold else None
        for bi, match in matches.items()
    }


class ImageInfo:
    def __init__(
        self,
        path_image: str,
        device: str | None = None,
        model_path: str | None = None,
    ):
        model = _load_model(str(_resolve_path(model_path or _DEFAULT_MODEL_NAME)))
        self._all_names = model.names
        self._boxes: Boxes = model(
            str(_resolve_path(path_image)), device=device, verbose=False
        )[0].boxes

    def _class_name(self, box) -> str:
        return self._all_names[box.cls.item()]

    def _center(self, index: int) -> tuple[float, float]:
        x_center, y_center, *_ = self._boxes[index].xywhn[0].tolist()
        return x_center, y_center

    @cached_property
    def _class_indices(self) -> dict[str, list[int]]:
        indices: dict[str, list[int]] = {}
        for bi, box in enumerate(self._boxes):
            indices.setdefault(self._class_name(box), []).append(bi)
        return indices

    @cached_property
    def _belongings_person(self) -> dict[int, tuple[int, float] | None]:
        belonging_indices = self.boxes_class("suitcase") + self.boxes_class("handbag")
        person_indices = self.boxes_class("person")
        return {
            bi: _min_distance_index(bi, person_indices, self._center)
            for bi in belonging_indices
        }

    @cached_property
    def _data_frame(self) -> DataFrame:
        xyxy = self._boxes.xyxy.cpu().numpy()
        cls = self._boxes.cls.cpu().numpy()
        conf = self._boxes.conf.cpu().numpy()
        df = DataFrame(xyxy, columns=["xmin", "ymin", "xmax", "ymax"])
        df["name"] = [self._all_names[c] for c in cls]
        df["confidence"] = conf
        return df

    def boxes_class(self, class_name: str) -> list[int]:
        return self._class_indices.get(class_name, [])

    def box_info(self, index: int) -> BoxInfo:
        box = self._boxes[index]
        xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
        return BoxInfo(xmin, ymin, xmax, ymax, box.conf.item(), self._class_name(box))

    def data_frame(self) -> DataFrame:
        return self._data_frame

    def suitcase_handbag_person(
        self, threshold: float
    ) -> dict[int, tuple[int, float] | None]:
        return _apply_threshold(self._belongings_person, threshold)
