from __future__ import annotations
from packages.license_plate_recognition.algorithm import LicensePlateRecognitionPipe
from packages.license_plate_recognition.data import LicensePlateRecognitionDataPipe
from src.package.package import Package
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from src.pipe import Pipe

class LicensePlateRecognitionPackage(Package):
    name: str = "License Plate Recognition"
    _pipes: Iterable[type[Pipe]] = [LicensePlateRecognitionPipe, LicensePlateRecognitionDataPipe]
    dependencies: dict[str, Package] = {}
