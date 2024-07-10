from __future__ import annotations
import src.pipe as pipe
import packages.license_plate_recognition.data as license_plate_recognition
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.manager import Manager
    from src.ui.common import ConfigPage

class LicensePlateRecognitionDataPipe(pipe.Pipe):
    cls_name: str = "License Plate Recognition (Data)"
    cls_config: type[pipe.Config] = license_plate_recognition.LicensePlateRecognitionDataConfig
    cls_function: type[pipe.Function] = license_plate_recognition.LicensePlateRecognitionDataFunction

    def __init__(self, name: str, manager: Manager, config: license_plate_recognition.LicensePlateRecognitionDataConfig) -> None:
        super().__init__(name, manager, config)
        self.config: license_plate_recognition.LicensePlateRecognitionDataConfig = config

    def config_ui(self, manager: Manager, config_page: ConfigPage) -> license_plate_recognition.LicensePlateRecognitionDataConfigUI:
        return license_plate_recognition.LicensePlateRecognitionDataConfigUI(self, manager, config_page)

    def play(self, manager: Manager) -> None:
        if self.playing:
            return
        self.playing = True

    def stop(self, manager: Manager, result: license_plate_recognition.LicensePlateRecognitionDataOutput) -> None:
        if not self.playing:
            return
        self.playing = False
        if result:
            manager.data[self.config.name] = result
