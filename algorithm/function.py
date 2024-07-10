from __future__ import annotations
from ultralytics import YOLO
from datetime import datetime, timedelta
import numpy as np
import json
import cv2
import easyocr
import torch
import rapidfuzz
from packages.license_plate_recognition.algorithm import LicensePlateRecognitionConfig
from packages.license_plate_recognition.common import double_replace
from src.pipe.function import IO, Function
from typing import Any, NamedTuple

class LicensePlateRecognitionInput(IO):
    frame: np.ndarray # = np.array([[[0, 0, 0]]])

class LicensePlateResult(NamedTuple):
    label: str # = ""
    license_plate: str # = ""
    known: bool # = ""
    detection_score: float # = 0.0
    ocr_score: float # = 0.0
    similarity_score: float # = 0.0
    box: tuple[int, int, int, int] # = (0, 0, 0, 0)

class BytesOutput(IO):
    data: bytes # = b""

class LicensePlateRecognitionFunction(Function):
    cls_input: type[LicensePlateRecognitionInput] = LicensePlateRecognitionInput

    def __init__(self, config: LicensePlateRecognitionConfig) -> None:
        self.config: LicensePlateRecognitionConfig = config
        self.license_plate_detector: YOLO = YOLO(self.config.path_to_models + "license_plate_detector.pt")
        if torch.cuda.is_available():
            self.license_plate_detector.to('cuda')
            self.reader: easyocr.Reader = easyocr.Reader(['en'], model_storage_directory=self.config.path_to_models, gpu=True)
        else:
            self.reader: easyocr.Reader = easyocr.Reader(['en'], model_storage_directory=self.config.path_to_models, gpu=False)

        if self.config.data:
            self.names: list[str] = self.config.data.names
            self.plates: list[str] = self.config.data.plates
            self.no_data: bool = False
        else:
            self.no_data: bool = True
        self.latest: dict[str, datetime] = {}
        self.frame_output = self.config.frame_output.cls_function(self.config.frame_output.config) if self.config.frame_output else None
        self.results_output = self.config.results_output.cls_function(self.config.results_output.config) if self.config.results_output else None
        self.attendance_output = self.config.attendance_output.cls_function(self.config.attendance_output.config) if self.config.attendance_output else None

    def __call__(self, input: LicensePlateRecognitionInput) -> IO:
        license_detections = self.license_plate_detector(input.frame.copy(), verbose=False)[0]
        res: list[LicensePlateResult] = []
        if len(license_detections.boxes.cls.tolist()) != 0:
            expand_x: int = self.config.expand_x
            expand_y: int = self.config.expand_y
            ocr_config: dict[str, Any] = self.config.ocr_config._asdict()
            for license_plate in license_detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                ocr_scores: float = 0.0
                plate: list[str] = []
                cropped = cv2.cvtColor(input.frame[int(y1) - expand_y:int(y2) + expand_y, int(x1) - expand_x: int(x2) + expand_x, :], cv2.COLOR_BGR2GRAY)

                detections = self.reader.readtext(cropped, **ocr_config)
                if detections:
                    box_size: int = cropped.shape[0]*cropped.shape[1]
                    sorted_detections = sorted( detections, key=lambda result:(int(result[0][0][1]/cropped.shape[1]/self.config.sorting_tolerance), result[0][0][0]) )

                    for (box, text, det_score) in sorted_detections:
                        if np.sum(np.subtract(box[1], box[0]))*np.sum(np.subtract(box[2], box[1])) / box_size > self.config.min_text_percentage:
                            ocr_scores += det_score
                            plate.append(text.upper())
                license_plate_text: str = "".join(plate)
                if len(plate) != 0 : 
                    license_plate_text_score: float = ocr_scores/len(plate)
                else :
                    license_plate_text_score: float = 0.0

                label: str = ""
                known: bool = False
                similarity: float = 0.0
                if license_plate_text:
                    license_plate_text: str = str(license_plate_text)
                    if not self.no_data:
                        fuzzed = rapidfuzz.process.extractOne(double_replace(license_plate_text), self.plates)
                        similarity: float = fuzzed[1]
                        if fuzzed[1] > self.config.similarity:
                            label: str = self.names[fuzzed[2]]
                            known: bool = True
                        else:
                            label: str = license_plate_text
                    else:
                        label: str = license_plate_text
                else:
                    license_plate_text: str = ""

                res.append(LicensePlateResult(
                    label=label,
                    license_plate=license_plate_text,
                    known=known,
                    detection_score=score,
                    ocr_score=license_plate_text_score,
                    similarity_score=similarity,
                    box=(x1, y1, x2, y2),
                ))
        
        if self.frame_output:
            if self.config.annotate:
                padding = self.config.annotations.padding
                half_padding = padding // 2
                for license_plate in res:
                    x1, y1, x2, y2 = license_plate.box
                    cv2.rectangle(input.frame, (x1, y1), (x2, y2), self.config.annotations.known_box_color if license_plate.known else self.config.annotations.unknown_box_color, self.config.annotations.box_thickness)
                    if label:
                        (text_width, text_height) = cv2.getTextSize(license_plate.label, self.config.annotations.font, self.config.annotations.text_scale, self.config.annotations.text_thickness)[0]
                        cv2.rectangle(input.frame, (x1, y1 - padding - text_height), (x1 + padding + text_width, y1), (255, 255, 255), cv2.FILLED)
                        cv2.putText(input.frame, license_plate.label, (x1 + half_padding, y1 - half_padding), self.config.annotations.font, self.config.annotations.text_scale, (0, 0, 0), self.config.annotations.text_thickness)

            flag, enc = cv2.imencode('.jpg', input.frame)
            if flag:
                self.frame_output(BytesOutput(
                    data=enc.tobytes(),
                ))

        if res:
            if self.results_output:
                self.results_output(BytesOutput(
                    data=(json.dumps({str(datetime.now()): [license_plate._asdict() for license_plate in res]}) + "\n").encode("latin-1"),
                ))

            if not self.no_data and self.attendance_output:
                names: list[str] = []
                now: datetime = datetime.now()
                for license_plate in res:
                    if license_plate.known:
                        name: str = license_plate.label
                        if (name not in self.latest) or (now - self.latest[name] > timedelta(seconds=self.config.attendance_interval)):
                            names.append(name)
                        self.latest[name] = now

                if names:
                    # self.attendance_output("".join(f"{now},{name}\n" for name in names).encode("latin-1"))
                    self.attendance_output(BytesOutput(
                        data="".join(["{\"", str(now), "\": ",  str(names).replace("'", '"'), "}\n"]).encode("latin-1"),
                    ))
        return IO()
