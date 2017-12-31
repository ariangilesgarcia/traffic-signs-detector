import os
import cv2
import json

from cropper import Cropper
from localizer import Localizer
from classifier import Classifier
from detection_pipeline import DetectionPipeline
from plotter import Plotter
from logger import Logger


class Detector:

    def __init__(self):
        # Create objects
        cropper = Cropper(0.25, force_square=True)
        localizer = Localizer(cfg_path='../data/yolo/full/trafficsigns.cfg',
                              weights_path='../data/yolo/full/trafficsigns.weights',
                              threshold=0.1)
        classifier = Classifier(model_path='../data/classifier/trafficsigns.json',
                                weights_path='../data/classifier/trafficsigns.h5',
                                labels_path='../data/classifier/classes.txt',
                                threshold=0.9)
        self.detection_pipeline = DetectionPipeline(localizer, cropper, classifier)
        self.plotter = Plotter(num_classes=20, bgr=True)
        self.logger = Logger()


    def detect_image(self,
                     image,
                     output=None):

        detections = self.detection_pipeline.detect_objects_in_image(image)

        if output:
            filename, extension = os.path.splitext(output)

            if extension == '.json':
                self.logger.save_detections_to_json(detections, output)
            elif extension == '.txt':
                h, w, _ = image.shape
                self.logger.save_detections_to_yolo(detections, (w, h), output)
            elif extension == '.csv':
                self.logger.save_detections_to_csv(detections, output)
            elif extension == '.jpg':
                detections_image = self.plotter.plot_detections(image, detections)
                cv2.imwrite(output, detections_image)

        return detections


if __name__ == '__main__':
    detector = Detector()
    img = cv2.imread('/home/arian/fortnite.png')
    detections = detector.detect_image(img, output='/home/arian/predictions.jpg')
