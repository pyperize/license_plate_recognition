# License Plate Recognition pipe for pyperize
Using YOLOv8, EasyOCR and RapidFuzz

### Prerequisites
- CUDA 12.4

## Install

1. Copy this package into ```./packages/```
2. Edit ```./packages/__init__.py``` to import the package
3. Add the package name and instance to the ```PACKAGES``` global variable in ```./packages/__init__.py```

```./packages/__init__.py``` should contain something like this where ```...``` are the other packages

```
from src.package import Package
from packages import (
    ...
    license_plate_recognition,
    ...
)

PACKAGES: dict[str, Package] = {
    ...
    license_plate_recognition.LicensePlateRecognitionPackage.name: license_plate_recognition.LicensePlateRecognitionPackage(),
    ...
}
```
