from __future__ import annotations
import json
import numpy as np
from packages.license_plate_recognition.common import reordered_plates
from src.pipe.function import IO, Function
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from packages.license_plate_recognition.data import LicensePlateRecognitionDataConfig

class LicensePlateRecognitionDataOutput(IO):
    name: str = "License Plate Recognition Data"
    names: list[str] = []
    plates: list[str] = []

class LicensePlateRecognitionDataFunction(Function):
    cls_output: type[LicensePlateRecognitionDataOutput] = LicensePlateRecognitionDataOutput

    def __init__(self, config: LicensePlateRecognitionDataConfig) -> None:
        self.config: LicensePlateRecognitionDataConfig = config

    def get_biggest_face(self, faces: list) -> int:
        biggest: int = 0
        max_area: float = (faces[0].bbox[2] - faces[0].bbox[0]) * (faces[0].bbox[3] - faces[0].bbox[1])
        for count in range(1, len(faces)):
            face = faces[count]
            area: float = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
            if area > max_area:
                max_area = area
                biggest = count
        return biggest

    def __call__(self, input: IO = IO()) -> LicensePlateRecognitionDataOutput:
        plates_to_names = reordered_plates(json.load(open(self.config.db_path, "r")))
        names: list[str] = []
        plates: list[str] = []

        for plate, name in plates_to_names.items():
            names.append(name)
            plates.append(plate)

        return LicensePlateRecognitionDataOutput(
            name=self.config.name,
            names=names,
            plates=plates,
        )
