import cv2
import pytest

from .context import detector
from detector.classifier import Classifier


class TestClassifier:

    # TC-CSF-01
    def test_classifier(self):
        model_path = './data/classifier/trafficsigns.json'
        weights_path = './data/classifier/trafficsigns.h5'
        labels_path = './data/classifier/classes.txt'
        threshold = 0.5

        classifier = Classifier(model_path, weights_path, labels_path, threshold)

        img = cv2.imread('./tests/data/test_sq.png')

        prediction = classifier.classify_image(img)

        assert prediction['class_id'] == 15


    # TC-CSF-02
    def test_set_threshold(self):
        model_path = './data/classifier/trafficsigns.json'
        weights_path = './data/classifier/trafficsigns.h5'
        labels_path = './data/classifier/classes.txt'
        threshold = 0.5

        classifier = Classifier(model_path, weights_path, labels_path, threshold)

        classifier.set_threshold(0.9)


        new_threshold = classifier.get_threshold()

        assert new_threshold == 0.9


    # TC-CSF-03
    def test_out_of_bounds_threshold(self):
        model_path = './data/classifier/trafficsigns.json'
        weights_path = './data/classifier/trafficsigns.h5'
        labels_path = './data/classifier/classes.txt'
        threshold = 0.5

        classifier = Classifier(model_path, weights_path, labels_path, threshold)

        with pytest.raises(Exception):
            classifier.set_threshold(1.5)
