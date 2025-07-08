# This checks that all classes defined in the `detector.py` module inherit from the `Detector` base class.

import inspect
import tree_detection_framework.detection.detector as detector_module
from tree_detection_framework.detection.detector import Detector

def test_all_detectors_inherit_from_Detector():
    # Get all classes defined in detector.py
    all_classes = inspect.getmembers(detector_module, inspect.isclass)

    # Check each class except the base `Detector`
    for name, cls in all_classes:
        if cls.__module__ != detector_module.__name__:
            continue  # Skip classes imported to detector.py
        if name is Detector:
            continue
        assert issubclass(cls, Detector), f"{name} does not inherit from Detector"