import cv2

from .context import detector
from detector.cropper import Cropper
from detector.localizer import Localizer
from detector.classifier import Classifier
from detector.detection_pipeline import DetectionPipeline


class TestDetectionPipeline:

    def test_detector_pipeline(self):
        # Create Localizer
        cfg_path = './tests/data/trafficsigns.cfg'
        weights_path = './tests/data/trafficsigns.weights'
        threshold = 0.24

        localizer = Localizer(cfg_path, weights_path, threshold, gpu=0.0)

        # Create cropper
        crop_percent = 0.25
        force_square = True

        cropper = Cropper(crop_percent, force_square)

        # Create classifier
        model_path = './tests/data/trafficsigns.json'
        weights_path = './tests/data/trafficsigns.h5'
        labels_path = './tests/data/classes.txt'
        threshold = 0.5

        classifier = Classifier(model_path, weights_path, labels_path, threshold)

        # Create detection pipeline
        pipeline = DetectionPipeline(localizer, cropper, classifier)


        # Detect on image
        img = cv2.imread('./tests/data/test.png')

        detections = pipeline.detect_objects_in_image(img)

        print(detections)



        assert 0 == 0
