import cv2

from .context import detector
from detector.localizer import Localizer


class TestLocalizer:

    def test_localizer(self):
        cfg_path = './data/yolo/full/trafficsigns.cfg'
        weights_path = './data/yolo/full/trafficsigns.weights'
        threshold = 0.24

        localizer = Localizer(cfg_path, weights_path, threshold, gpu=0.0)

        img = cv2.imread('./tests/data/test.png')

        true_roi = [
            {
                'label': 'traffic-sign',
                'class_id': 0,
                'coordinates': [1449, 476, 1575, 592],
                'confidence': 0.8594279
            }
        ]
        predicted_roi = localizer.find_objects_in_image(img)

        assert true_roi[0]['coordinates'] == predicted_roi[0]['coordinates']
