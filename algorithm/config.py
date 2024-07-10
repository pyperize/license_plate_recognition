from __future__ import annotations
import flet as ft
import cv2
from src.pipe import Pipe, Config, ConfigUI
from src.ui.pipe.tile import PipeTile
from typing import Any, Callable, NamedTuple, TYPE_CHECKING
from packages.license_plate_recognition.data import LicensePlateRecognitionDataOutput
if TYPE_CHECKING:
    from src.manager import Manager
    from src.ui.common import ConfigPage
    from packages.license_plate_recognition.algorithm import LicensePlateRecognitionPipe

class EasyOCRConfig(NamedTuple):
    decoder: str = "beamsearch"
    allowlist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    beamWidth: int = 8
    batch_size: int = 64
    text_threshold: float = 0.70
    low_text: float = 0.40
    link_threshold: float = 0.40

class AnnotationConfig(NamedTuple):
    padding: int = 20
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    text_scale: float = 0.5
    text_thickness: int = 1
    known_box_color: tuple[int, int, int] = (0, 255, 0)
    unknown_box_color: tuple[int, int, int] = (0, 0, 255)
    box_thickness: int = 2

class LicensePlateRecognitionConfig(Config):
    frame_output: Pipe | None = None
    results_output: Pipe | None = None
    attendance_output: Pipe | None = None
    data: LicensePlateRecognitionDataOutput | None = None
    path_to_models: str = "./packages/license_plate_recognition/models/"
    sorting_tolerance: float = 0.33
    min_text_percentage: float = 0.14
    expand_x: int = 0
    expand_y: int = 0
    similarity: int = 90
    attendance_interval: int = 300
    annotate: bool = True
    annotations: AnnotationConfig = AnnotationConfig()
    ocr_config: EasyOCRConfig = EasyOCRConfig()

class LicensePlateRecognitionConfigUI(ConfigUI):
    def __init__(self, instance: LicensePlateRecognitionPipe, manager: Manager, config_page: ConfigPage) -> None:
        super().__init__(instance, manager, config_page)
        self.instance: LicensePlateRecognitionPipe = instance
        self.data_options: list[ft.dropdown.Option] = []
        self.refresh_data_options(False)
        self.content: ft.Column = ft.Column([
            PipeTile(
                "Frame",
                self.manager,
                self.config_page,
                self.select_pipe,
                self.delete_pipe(0),
                self.instance.config.frame_output,
            ),
            PipeTile(
                "Results",
                self.manager,
                self.config_page,
                self.select_pipe,
                self.delete_pipe(1),
                self.instance.config.results_output,
            ),
            PipeTile(
                "Attendance",
                self.manager,
                self.config_page,
                self.select_pipe,
                self.delete_pipe(2),
                self.instance.config.attendance_output,
            ),
            ft.Dropdown(
                self.instance.config.data.name if self.instance.config.data else None,
                label="Data source to use",
                hint_text="Data source to use",
                options=self.data_options,
                on_click=self.refresh_data_options,
                border_color="grey",
                dense=True,
            ),
            ft.TextField(self.instance.config.path_to_models, label="Path to Models", border_color="grey"),
            ft.TextField(str(self.instance.config.expand_x), label="Expand Crop (Width)", border_color="grey", input_filter=ft.NumbersOnlyInputFilter()),
            ft.TextField(str(self.instance.config.expand_y), label="Expand Crop (Height)", border_color="grey", input_filter=ft.NumbersOnlyInputFilter()),
            ft.TextField(str(self.instance.config.min_text_percentage), label="Minimum Text Percentage", border_color="grey", input_filter=ft.InputFilter(allow=True, regex_string=r"[0-9\.]", replacement_string="")),
            ft.TextField(str(self.instance.config.sorting_tolerance), label="Text Sorting Tolerance", border_color="grey", input_filter=ft.InputFilter(allow=True, regex_string=r"[0-9\.]", replacement_string="")),
            ft.TextField(str(self.instance.config.similarity), label="Text Similarity Threshold", border_color="grey", input_filter=ft.NumbersOnlyInputFilter()),
            ft.TextField(str(self.instance.config.attendance_interval), label="Interval after Latest Detection", border_color="grey", input_filter=ft.NumbersOnlyInputFilter()),
            ft.Switch(label="Annotate Output Frames", value=self.instance.config.annotate),
            ft.Container(
                ft.Text("Annotation Configuration"),
                padding=ft.padding.symmetric(vertical=20),
            ),
            ft.TextField("%02x%02x%02x" % self.instance.config.annotations.known_box_color, label="Known Box Color (hex)", border_color="grey"),
            ft.TextField("%02x%02x%02x" % self.instance.config.annotations.unknown_box_color, label="Unknown Box Color (hex)", border_color="grey"),
            ft.TextField(str(self.instance.config.annotations.box_thickness), label="Box Thickness", border_color="grey", input_filter=ft.NumbersOnlyInputFilter()),
            ft.TextField(str(self.instance.config.annotations.text_scale), label="Text Scale", border_color="grey", input_filter=ft.InputFilter(allow=True, regex_string=r"[0-9\.]", replacement_string="")),
            ft.TextField(str(self.instance.config.annotations.text_thickness), label="Text Thickness", border_color="grey", input_filter=ft.NumbersOnlyInputFilter()),
            ft.TextField(str(self.instance.config.annotations.padding), label="Padding", border_color="grey", input_filter=ft.NumbersOnlyInputFilter()),
            ft.Container(
                ft.Text("OCR Parameters"),
                padding=ft.padding.symmetric(vertical=20),
            ),
            ft.TextField(self.instance.config.ocr_config.decoder, label="Decoder", border_color="grey"),
            ft.TextField(self.instance.config.ocr_config.allowlist, label="Allowed Characters", border_color="grey"),
            ft.TextField(str(self.instance.config.ocr_config.beamWidth), label="Beam Width", border_color="grey", input_filter=ft.NumbersOnlyInputFilter()),
            ft.TextField(str(self.instance.config.ocr_config.batch_size), label="Batch Size", border_color="grey", input_filter=ft.NumbersOnlyInputFilter()),
            ft.TextField(str(self.instance.config.ocr_config.text_threshold), label="Text Threshold", border_color="grey", input_filter=ft.InputFilter(allow=True, regex_string=r"[0-9\.]", replacement_string="")),
            ft.TextField(str(self.instance.config.ocr_config.low_text), label="Text Low-Bound Score", border_color="grey", input_filter=ft.InputFilter(allow=True, regex_string=r"[0-9\.]", replacement_string="")),
            ft.TextField(str(self.instance.config.ocr_config.link_threshold), label="Link Threshold", border_color="grey", input_filter=ft.InputFilter(allow=True, regex_string=r"[0-9\.]", replacement_string="")),
        ])

    def refresh_data_options(self, update: bool = True):
        self.data_options.clear()
        self.data_options.extend([ft.dropdown.Option(id) for id in self.manager.data])
        if update:
            self.update()

    def select_pipe(self, cls: type[Pipe] | Pipe) -> Pipe:
        if isinstance(cls, type):
            cls: Pipe = cls(cls.cls_name, self.manager, cls.cls_config())
        return cls

    def delete_pipe(self, index: int) -> Callable[[Any], None]:
        def _delete_pipe(e) -> None:
            self.content.controls[index].pipe_selector.value = None
            self.content.controls[index].select_changed(None)
            self.update()
        return _delete_pipe

    def dismiss(self) -> None:
        self.instance.config = LicensePlateRecognitionConfig(
            frame_output=self.content.controls[0].instance,
            results_output=self.content.controls[1].instance,
            attendance_output=self.content.controls[2].instance,
            data=self.manager.data[self.content.controls[3].value] if self.content.controls[3].value else None,
            path_to_models=self.content.controls[4].value,
            expand_x=int(self.content.controls[5].value),
            expand_y=int(self.content.controls[6].value),
            min_text_percentage=float(self.content.controls[7].value),
            sorting_tolerance=float(self.content.controls[8].value),
            similarity=int(self.content.controls[9].value),
            attendance_interval=int(self.content.controls[10].value),
            annotate=self.content.controls[11].value,
            annotations=AnnotationConfig(
                known_box_color=tuple(bytes.fromhex(self.content.controls[13].value)),
                unknown_box_color=tuple(bytes.fromhex(self.content.controls[14].value)),
                box_thickness=int(self.content.controls[15].value),
                text_scale=float(self.content.controls[16].value),
                text_thickness=int(self.content.controls[17].value),
                padding=int(self.content.controls[18].value),
            ),
            ocr_config=EasyOCRConfig(
                decoder=self.content.controls[20].value,
                allowlist=self.content.controls[21].value,
                beamWidth=int(self.content.controls[22].value),
                batch_size=int(self.content.controls[23].value),
                text_threshold=float(self.content.controls[24].value),
                low_text=float(self.content.controls[25].value),
                link_threshold=float(self.content.controls[26].value),
            ),
        )
