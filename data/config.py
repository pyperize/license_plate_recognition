from __future__ import annotations
import flet as ft
from src.pipe.config import Config, ConfigUI
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.manager import Manager
    from src.ui.common import ConfigPage
    from packages.license_plate_recognition.data import LicensePlateRecognitionDataPipe

class LicensePlateRecognitionDataConfig(Config):
    name: str = "License Plate Recognition Data"
    db_path: str = "./../../guests.json"

class LicensePlateRecognitionDataConfigUI(ConfigUI):
    def __init__(self, instance: LicensePlateRecognitionDataPipe, manager: Manager, config_page: ConfigPage, content: ft.Control | None = None) -> None:
        self.instance: LicensePlateRecognitionDataPipe = instance
        super().__init__(instance, manager, config_page, ft.Column(spacing=20, controls=[
            ft.TextField(self.instance.config.name, label="Unique Database Name", border_color="grey"),
            ft.TextField(self.instance.config.db_path, label="Database Path", border_color="grey"),
        ]))

    def dismiss(self) -> None:
        self.instance.config = LicensePlateRecognitionDataConfig(
            name=self.content.controls[0].value,
            db_path=self.content.controls[1].value,
        )
