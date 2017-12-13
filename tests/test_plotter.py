import cv2
import unittest
import numpy as np
import urllib.request

from .context import detector
from detector.plotter import Plotter


class TestPlotter:


    def test_plot_detections(self):
        plt = Plotter(num_classes=20)

        test_detections = [
            {
                'coordinates': [200, 100, 300, 200],
                'confidence': 0.998,
                'class_id': 16,
                'label': 'contramano'
            },
            {
                'coordinates': [300, 800, 400, 950],
                'confidence': 0.783,
                'class_id': 13,
                'label': 'cocoa'
            }
        ]

        test_image_url = 'https://www.pyimagesearch.com/wp-content/uploads/2015/01/opencv_logo.png'
        response = urllib.request.urlopen(test_image_url)
        image = np.asarray(bytearray(response.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        drawn_image = plt.plot_detections(image, test_detections)

        assert drawn_image.shape == image.shape
