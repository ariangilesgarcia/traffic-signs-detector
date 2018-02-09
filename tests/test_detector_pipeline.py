import cv2

from .context import detector
from detector.cropper import Cropper
from detector.localizer import Localizer
from detector.classifier import Classifier
from detector.detection_pipeline import DetectionPipeline


class TestDetectionPipeline:

    def test_detector_pipeline(self):
        # Create Localizer
        cfg_path = './data/yolo/full/trafficsigns.cfg'
        weights_path = './data/yolo/full/trafficsigns.weights'
        threshold = 0.24

        localizer = Localizer(cfg_path, weights_path, threshold, gpu=0.0)

        # Create cropper
        crop_percent = 0.25
        force_square = True

        cropper = Cropper(crop_percent, force_square)

        # Create classifier
        model_path = './data/classifier/trafficsigns.json'
        weights_path = '/data/classifier/trafficsigns.h5'
        labels_path = '/data/classifier/classes.txt'
        threshold = 0.5

        classifier = Classifier(model_path, weights_path, labels_path, threshold)

        # Create detection pipeline
        pipeline = DetectionPipeline(localizer, cropper, classifier)


        # Detect on image
        img = cv2.imread('./tests/data/test.png')

        detections = pipeline.detect_objects_in_image(img)

        print(detections)



        assert 0 == 0
