import cv2
import pytest

from .context import detector
from detector.classifier import Classifier


class TestClassifier:

    def test_classifier(self):
        model_path = './tests/data/trafficsigns.json'
        weights_path = './tests/data/trafficsigns.h5'
        labels_path = './tests/data/classes.txt'
        threshold = 0.5

        classifier = Classifier(model_path, weights_path, labels_path, threshold)

        img = cv2.imread('./tests/data/test_sq.png')

        prediction = classifier.classify_image(img)

        assert prediction['class_id'] == 15


    def test_set_threshold(self):
        model_path = './tests/data/trafficsigns.json'
        weights_path = './tests/data/trafficsigns.h5'
        labels_path = './tests/data/classes.txt'
        threshold = 0.5

        classifier = Classifier(model_path, weights_path, labels_path, threshold)

        classifier.set_threshold(0.9)


        new_threshold = classifier.get_threshold()

        assert new_threshold == 0.9


    def test_out_of_bounds_threshold(self):
        model_path = './tests/data/trafficsigns.json'
        weights_path = './tests/data/trafficsigns.h5'
        labels_path = './tests/data/classes.txt'
        threshold = 0.5

        classifier = Classifier(model_path, weights_path, labels_path, threshold)

        with pytest.raises(Exception):
            classifier.set_threshold(1.5)
