import unittest
import numpy as np

from .context import detector
from detector.cross_frame_detector import CrossFrameDetector


class TestCrossFrameDetector:

    def test_register_detections(self):
        test_detections = [
            {
                'coordinates': [200, 100, 300, 200],
                'confidence': 0.998,
                'class_id': 0,
                'label': 'contramano'
            },
            {
                'coordinates': [300, 500, 800, 900],
                'confidence': 0.734,
                'class_id': 3,
                'label': 'cruce'
            },
        ]

        cfd = CrossFrameDetector(num_classes=4, frame_history_count=10, frames_threshold=5)
        cfd.register_detections(test_detections)

        assert cfd.detection_history[-1, 0] == 1 and cfd.detection_history[-1, 3] == 1


    def test_get_detections(self):
        test_detections = [
            {
                'coordinates': [200, 100, 300, 200],
                'confidence': 0.998,
                'class_id': 0,
                'label': 'contramano'
            },
            {
                'coordinates': [300, 500, 800, 900],
                'confidence': 0.734,
                'class_id': 3,
                'label': 'cruce'
            },
        ]

        cfd = CrossFrameDetector(num_classes=4, frame_history_count=10, frames_threshold=5)

        for _ in range(5):
            cfd.register_detections(test_detections)

        cfr_detections = cfd.get_detections()

        assert cfr_detections == [0, 3]
