import unittest

from .context import detector
from detector.utils import convert_detection_to_csv
from detector.utils import convert_detection_to_yolo
from detector.utils import convert_coordinates_to_relative


class TestUtils:


    def test_convert_to_csv(self):
        test_detection = {
            'coordinates': [200, 100, 300, 200],
            'confidence': 0.998,
            'class_id': 16,
            'label': 'contramano'
        }

        csv_output = convert_detection_to_csv(test_detection)
        expected_output = '16, 200, 100, 300, 200'

        assert csv_output == expected_output


    def test_convert_to_csv_with_label(self):
        test_detection = {
            'coordinates': [200, 100, 300, 200],
            'confidence': 0.998,
            'class_id': 16,
            'label': 'contramano'
        }

        csv_output = convert_detection_to_csv(test_detection, include_label=True)
        expected_output = '16, contramano, 200, 100, 300, 200'

        assert csv_output == expected_output


    def test_convert_to_csv_with_confidence(self):
        test_detection = {
            'coordinates': [200, 100, 300, 200],
            'confidence': 0.998,
            'class_id': 16,
            'label': 'contramano'
        }

        csv_output = convert_detection_to_csv(test_detection, include_confidence=True)
        expected_output = '16, 0.998, 200, 100, 300, 200'

        assert csv_output == expected_output


    def test_convert_to_csv_with_label_confidence(self):
        test_detection = {
            'coordinates': [200, 100, 300, 200],
            'confidence': 0.998,
            'class_id': 16,
            'label': 'contramano'
        }

        csv_output = convert_detection_to_csv(test_detection, include_label=True, include_confidence=True)
        expected_output = '16, contramano, 0.998, 200, 100, 300, 200'

        assert csv_output == expected_output


    def test_convert_to_yolo(self):
        test_detection = {
            'coordinates': [200, 100, 300, 200],
            'confidence': 0.998,
            'class_id': 16,
            'label': 'contramano'
        }

        test_image_size = (500, 500)

        yolo_output = convert_detection_to_yolo(test_detection, test_image_size)
        expected_output = '16 0.5 0.3 0.2 0.2'

        assert yolo_output == expected_output


    def test_convert_yo_relative(self):
        test_bbox = (100, 100, 200, 200)
        test_image_size = (1000, 1000)

        relative_coordinates = convert_coordinates_to_relative(test_bbox, test_image_size)
        expected_coordiantes = (0.15, 0.15, 0.1, 0.1)

        assert relative_coordinates == expected_coordiantes
