from __future__ import annotations
import src.pipe as pipe
import packages.license_plate_recognition.algorithm as license_plate_recognition
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.manager import Manager
    from src.ui.common import ConfigPage

class LicensePlateRecognitionPipe(pipe.Pipe):
    cls_name: str = "License Plate Recognition"
    cls_config: type[license_plate_recognition.LicensePlateRecognitionConfig] = license_plate_recognition.LicensePlateRecognitionConfig
    cls_function: type[license_plate_recognition.LicensePlateRecognitionFunction] = license_plate_recognition.LicensePlateRecognitionFunction

    def __init__(self, name: str, manager: Manager, config: license_plate_recognition.LicensePlateRecognitionConfig | None = None) -> None:
        super().__init__(name, manager, config)
        self.config: license_plate_recognition.LicensePlateRecognitionConfig = config if config else self.cls_config()

    def config_ui(self, manager: Manager, config_page: ConfigPage) -> license_plate_recognition.LicensePlateRecognitionConfigUI:
        return license_plate_recognition.LicensePlateRecognitionConfigUI(self, manager, config_page)

    def play(self, manager: Manager) -> None:
        if self.playing:
            return
        self.playing = True
        if self.config.frame_output: self.config.frame_output.play(manager)
        if self.config.results_output: self.config.results_output.play(manager)
        if self.config.attendance_output: self.config.attendance_output.play(manager)

    def stop(self, manager: Manager, result: pipe.IO) -> None:
        if not self.playing:
            return
        self.playing = False
        if self.config.frame_output: self.config.frame_output.stop(manager, result)
        if self.config.results_output: self.config.results_output.stop(manager, result)
        if self.config.attendance_output: self.config.attendance_output.stop(manager, result)
