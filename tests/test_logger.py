import json

from .context import detector
from detector.logger import Logger


class TestLogger:

    def test_save_detections_csv(self):
        test_detection = [
            {
                'coordinates': [200, 100, 300, 200],
                'confidence': 0.998,
                'class_id': 16,
                'label': 'contramano'
            }
        ]

        log = Logger()

        output_filename = './tests/data/output.csv'
        log.save_detections_to_csv(test_detection, output_filename)

        expected_output = '16, 200, 100, 300, 200'
        csv_content = open(output_filename, 'r').read().strip()

        assert csv_content == expected_output


    def test_save_detections_yolo(self):
        test_detection = [
            {
                'coordinates': [200, 100, 300, 200],
                'confidence': 0.998,
                'class_id': 16,
                'label': 'contramano'
            }
        ]

        log = Logger()

        output_filename = './tests/data/output.txt'
        image_size = (1000, 1000)

        log.save_detections_to_yolo(test_detection, image_size, output_filename)

        expected_output = '16 0.25 0.15 0.1 0.1'
        yolo_content = open(output_filename, 'r').read().strip()

        assert yolo_content == expected_output


    def test_save_detections_json(self):
        test_detection = [
            {
                'coordinates': [200, 100, 300, 200],
                'confidence': 0.998,
                'class_id': 16,
                'label': 'contramano'
            }
        ]


        log = Logger()

        output_filename = './tests/data/output.json'

        log.save_detections_to_json(test_detection, output_filename)

        with open(output_filename, 'r') as fp:
            saved_json = json.load(fp)

        assert test_detection == saved_json
