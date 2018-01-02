import os
import cv2
import glob
import json
import numpy as np

from logger import Logger
from plotter import Plotter
from cropper import Cropper
from localizer import Localizer
from classifier import Classifier
from detection_pipeline import DetectionPipeline
from cross_frame_detector import CrossFrameDetector

from exceptions import FormatNotSupportedException


class Detector:

    def __init__(self, detection_pipeline):
        self.detection_pipeline = detection_pipeline
        self.plotter = Plotter(num_classes=20, bgr=True)
        self.logger = Logger()
        self.cfd = CrossFrameDetector(num_classes=20,
                                      frame_history_count=5,
                                      frames_threshold=2)

        self.classes_images = self.load_classes_images('../data/classifier/classes')


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
            else:
                raise FormatNotSupportedException('Output extension must be .json .txt .csv or .jpg')

        return detections


    def detect_video(self,
                     video_path,
                     show_output=False,
                     output=None):

        cap = cv2.VideoCapture(video_path)

        if show_output:
            from screeninfo import get_monitors
            monitor = get_monitors()[0]
            screen_w, screen_h, = monitor.width, monitor.height

            window_name = 'Detector'
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.moveWindow(window_name, screen_w - 1, screen_h - 1)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


    def detect_feed(self,
                    feed,
                    show_output=False,
                    output=None):

        cap = cv2.VideoCapture(feed)

        if show_output:
            from screeninfo import get_monitors
            monitor = get_monitors()[0]
            screen_w, screen_h, = monitor.width, monitor.height

            window_name = 'Detector'
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.moveWindow(window_name, screen_w - 1, screen_h - 1)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                break

            detections = self.detection_pipeline.detect_objects_in_image(frame)
            self.cfd.register_detections(detections)

            frame = self.plotter.plot_detections(frame, detections)

            cfd_detections = self.cfd.get_detections()

            cfd_frame = self.draw_cfd_detections(frame, cfd_detections)

            # Display the resulting frame
            if show_output:
                cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


    def load_classes_images(self, path):
        classes_images = {}

        images_list = glob.glob(os.path.join(path, '*.jpg'))

        for image_path in images_list:
            class_id = int(os.path.basename(image_path)[:-4])
            image = cv2.imread(image_path)
            image = cv2.copyMakeBorder(image,10,10,10,10,
                                      cv2.BORDER_CONSTANT,
                                      value=[0,0,0])

            classes_images[class_id] = image

        return classes_images


    def draw_cfd_detections(self, frame, cfd_detections):
        cfd_detection_count = len(cfd_detections)

        if cfd_detection_count > 0:
            frame_height, frame_width, _ = frame.shape
            class_image_size = int(frame_height/4)

            aux = np.zeros(shape=(class_image_size,
                                  class_image_size*cfd_detection_count,
                                  3),
                           dtype='uint8')

            start_h, start_w = 0, 0

            for class_id in cfd_detections:
                img = self.classes_images[class_id]
                img = cv2.resize(img, (class_image_size, class_image_size))

                end_w = start_w + img.shape[1]
                aux[start_h:img.shape[0], start_w:end_w] = img
                start_w = start_w + img.shape[1]

            frame[frame_height-class_image_size:, :class_image_size*cfd_detection_count] = aux

        return frame


if __name__ == '__main__':
    cropper = Cropper(0.25, force_square=True)
    localizer = Localizer(cfg_path='../data/yolo/full/trafficsigns.cfg',
                          weights_path='../data/yolo/full/trafficsigns.weights',
                          threshold=0.1)
    classifier = Classifier(model_path='../data/classifier/trafficsigns.json',
                            weights_path='../data/classifier/trafficsigns.h5',
                            labels_path='../data/classifier/classes.txt',
                            threshold=0.9)
    detection_pipeline = DetectionPipeline(localizer, cropper, classifier)

    detector = Detector(detection_pipeline)

    detections = detector.detect_feed(0, show_output=True)
