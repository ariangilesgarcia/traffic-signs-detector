import os
import cv2
import unittest
import numpy as np

from .context import detector
from detector.plotter import Plotter


class TestPlotter:

    # TC-PLT-01
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

        test_image_path = 'tests/data/test.jpg'
        image = cv2.imread(test_image_path)

        drawn_image = plt.plot_detections(image, test_detections)

        assert drawn_image.shape == image.shape


    # TC-PLT-02
    def test_save_plot_detections(self):
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

        test_image_path = 'tests/data/test.jpg'
        image = cv2.imread(test_image_path)

        output_filename = './tests/data/test_output.jpg'
        drawn_image = plt.plot_detections(image, test_detections, output=output_filename)

        assert os.path.exists(output_filename) == True


    # TC-PLT-03
    def test_plot_detections_bgr(self):
        plt = Plotter(num_classes=20, bgr=True)

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

        test_image_path = 'tests/data/test.jpg'
        image = cv2.imread(test_image_path)

        drawn_image = plt.plot_detections(image, test_detections)

        assert drawn_image.shape == image.shape

    # TC-PLT-04
    def test_plot_detections_confidence(self):
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

        test_image_path = 'tests/data/test.jpg'
        image = cv2.imread(test_image_path)

        drawn_image = plt.plot_detections(image, test_detections, draw_confidence=True)

        assert drawn_image.shape == image.shape


    # TC-PLT-05
    def test_plot_detections_right_side(self):
        plt = Plotter(num_classes=20)

        test_detections = [
            {
                'coordinates': [450, 400, 500, 500],
                'confidence': 0.998,
                'class_id': 16,
                'label': 'test-long-long-long-long-long-long-long-long-long-long-label'
            },
            {
                'coordinates': [300, 800, 400, 950],
                'confidence': 0.783,
                'class_id': 13,
                'label': 'cocoa'
            }
        ]

        test_image_path = 'tests/data/test.jpg'
        image = cv2.imread(test_image_path)

        drawn_image = plt.plot_detections(image, test_detections)

        assert drawn_image.shape == image.shape


    # TC-PLT-05
    def test_plot_detections_top_side(self):
        plt = Plotter(num_classes=20)

        test_detections = [
            {
                'coordinates': [0, 0, 250, 400],
                'confidence': 0.998,
                'class_id': 16,
                'label': 'contramano'
            },
        ]

        test_image_path = 'tests/data/test.jpg'
        image = cv2.imread(test_image_path)

        drawn_image = plt.plot_detections(image, test_detections)

        assert drawn_image.shape == image.shape
