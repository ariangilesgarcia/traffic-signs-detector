import os
import cv2

from .context import detector
from detector.cropper import Cropper
from detector.localizer import Localizer
from detector.classifier import Classifier
from detector.detection_pipeline import DetectionPipeline

from detector.detector import Detector
from detector.detector import create_detector_from_file


class TestDetector:

    def test_detector_image(self):
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
        detection_pipeline = DetectionPipeline(localizer, cropper, classifier)

        # Create detector
        images_path = '/data/classifier/classes/'
        sounds_path = '/data/sounds/'

        detector = Detector(detection_pipeline, images_path, sounds_path)

        # Detect on image
        img = cv2.imread('./tests/data/test.png')

        image, detections = detector.detect_image(img, show_confidence=True, return_image=True)

        true_detections = [
            {
                'class_id': 15,
                'coordinates': [1434, 456, 1590, 612],
                'label': 'max-60',
                'confidence': 1.0
            }
        ]

        assert detections == true_detections


    def test_detector_image_output_json(self):
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
        detection_pipeline = DetectionPipeline(localizer, cropper, classifier)

        # Create detector
        detector = Detector(detection_pipeline)

        # Detect on image
        img = cv2.imread('./tests/data/test.png')

        output_filename = './tests/data/detector_image.json'
        detections = detector.detect_image(img, output=output_filename)

        assert os.path.exists(output_filename) == True


    def test_detector_image_output_yolo(self):
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
        detection_pipeline = DetectionPipeline(localizer, cropper, classifier)

        # Create detector
        detector = Detector(detection_pipeline)

        # Detect on image
        img = cv2.imread('./tests/data/test.png')

        output_filename = './tests/data/detector_image.txt'
        detections = detector.detect_image(img, output=output_filename)

        assert os.path.exists(output_filename) == True


    def test_detector_image_output_csv(self):
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
        detection_pipeline = DetectionPipeline(localizer, cropper, classifier)

        # Create detector
        detector = Detector(detection_pipeline)

        # Detect on image
        img = cv2.imread('./tests/data/test.png')

        output_filename = './tests/data/detector_image.csv'
        detections = detector.detect_image(img, output=output_filename)

        assert os.path.exists(output_filename) == True


    def test_detector_image_output_jpg(self):
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
        detection_pipeline = DetectionPipeline(localizer, cropper, classifier)

        # Create detector
        detector = Detector(detection_pipeline)

        # Detect on image
        img = cv2.imread('./tests/data/test.png')

        output_filename = './tests/data/detector_image.jpg'
        detections = detector.detect_image(img, output=output_filename)

        assert os.path.exists(output_filename) == True


    def test_detector_video(self):
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
        detection_pipeline = DetectionPipeline(localizer, cropper, classifier)

        # Create detector
        detector = Detector(detection_pipeline)

        # Detect on video
        video_feed = './tests/data/test.mp4'

        output_mp4 = './tests/data/test_output.mp4'
        output_csv = './tests/data/test_output.csv'

        exit_code = detector.detect_video_feed(video_feed,
                                               output=output_mp4,
                                               output_csv=output_csv,
                                               show_confidence=True,
                                               sound_notifications=True)

        assert exit_code == True


    def test_detector_from_file(self):
        # Create detector
        detector = create_detector_from_file('./tests/data/config.json')

        assert isinstance(detector, Detector) == True
