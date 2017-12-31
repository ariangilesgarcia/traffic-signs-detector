import os
import cv2
import json

from detector.cropper import Cropper
from detector.localizer import Localizer
from detector.classifier import Classifier
from detector.detection_pipeline import DetectionPipeline
from detector.plotter import Plotter

from detector.utils import convert_detection_to_csv
from detector.utils import convert_detection_to_relative


class Detector:

    def __init__(self):
        # Create objects
        cropper = Cropper(0.25, force_square=True)
        localizer = Localizer(cfg_path='../../data/yolo/full/trafficsigns.cfg',
                              weights_path='../../data/yolo/full/trafficsigns.weights',
                              threshold=0.1)
        classifier = Classifier(model_path='../../data/classifier/trafficsigns.json',
                                weights_path='../../data/classifier/trafficsigns.h5',
                                labels_path='../../data/classifier/classes.txt',
                                threshold=0.9)
        self.detection_pipeline = DetectionPipeline(localizer, cropper, classifier)
        self.plotter = Plotter(num_classes=20, bgr=True)


    def detect_image(self,
                     image,
                     output=None,
                     save_format='json'):

        detections = self.detection_pipeline.detect_objects_in_image(image)

        if output:
            if save_format == 'json':
                with open(output, 'w') as fp:
                    json.dump(detections, fp)
            elif save_format == 'yolo':
                detections_yolo = convert_detection_to_relative(detections)
                with open(output, 'w') as fp:
                    fp.write(detections_yolo)
            elif save_format == 'csv':
                detections_csv = convert_detection_to_csv(detections)
                with open(output, 'w') as fp:
                    fp.write(detections_csv)
            elif save_format == 'image':
                detections_image = self.plotter.plot_detections(image, detections)
                cv2.imwrite(output, detections_image)
