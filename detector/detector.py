import os
import cv2
import glob
import json
import time
import math
import threading
import numpy as np

from detector.logger import Logger
from detector.plotter import Plotter
from detector.cropper import Cropper
from detector.localizer import Localizer
from detector.classifier import Classifier
from detector.detection_pipeline import DetectionPipeline
from detector.cross_frame_detector import CrossFrameDetector

from detector.exceptions import FormatNotSupportedException

from pydub import AudioSegment
from pydub.playback import play


class Detector:

    def __init__(self, detection_pipeline):
        self.detection_pipeline = detection_pipeline
        self.plotter = Plotter(num_classes=20, bgr=True)
        self.logger = Logger()
        self.cfd = CrossFrameDetector(num_classes=20,
                                      frame_history_count=5,
                                      frames_threshold=2)

        self.classes_images = self.load_classes_images('/home/arian/Projects/traffic-signs-detector/data/classifier/classes')

        self.notification_queue = []


    def detect_image(self,
                     image,
                     output=None,
                     show_confidence=True,
                     return_image=False):

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
                detections_image = self.plotter.plot_detections(image, detections, draw_confidence=show_confidence)
                cv2.imwrite(output, detections_image)
            else:
                raise FormatNotSupportedException('Output extension must be .json .txt .csv or .jpg')

        if return_image:
            detections_image = self.plotter.plot_detections(image, detections, draw_confidence=show_confidence)
            return detections_image, detections

        return detections


    def detect_video_feed(self,
                     video_feed,
                     show_output=False,
                     sound_notifications=False,
                     output=None,
                     output_csv=None):

        cap = cv2.VideoCapture(video_feed)
        #cap.set(cv2.CAP_PROP_FPS, 1)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if output:
            start = time.time()
            ret, frame = cap.read()
            detections = self.detection_pipeline.detect_objects_in_image(frame)
            end = time.time()

            elapsed = end - start
            fps = math.ceil(1/elapsed)

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output, fourcc, fps, (width, height))

        if output_csv:
            csv_file = open(output_csv, 'w')
            titles = ['frame_id', 'class_id', 'x1', 'y1', 'x2', 'y2']
            csv_file.write(', '.join(titles) + '\n')

        if show_output:
            from screeninfo import get_monitors
            monitor = get_monitors()[0]
            screen_w, screen_h, = monitor.width, monitor.height

            window_name = 'Detector'
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.moveWindow(window_name, screen_w - 1, screen_h - 1)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            if sound_notifications:
                sound_thread = threading.Thread(target=self.notification_thread, args=())
                sound_thread.daemon = True
                self.sound_notifications_lock = True
                sound_thread.start()

        frame_id = 0

        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame_id +=1

            if not ret:
                break

            detections = self.detection_pipeline.detect_objects_in_image(frame)

            if output_csv:
                for detection in detections:
                    class_id = detection['class_id']
                    coordinates = detection['coordinates']

                    line = ', '.join([str(x) for x in [frame_id, class_id, *coordinates]])
                    csv_file.write(line + '\n')

            self.cfd.register_detections(detections)

            frame = self.plotter.plot_detections(frame, detections)

            all_cfd_detections, current_cfd_detections = self.cfd.get_detections()

            if sound_notifications:
                for detection_id in current_cfd_detections:
                    self.notification_queue.append(detection_id)

            cfd_frame = self.draw_cfd_detections(frame, all_cfd_detections)

            # Display the resulting frame
            if show_output:
                cv2.imshow(window_name, frame)

            if output:
                out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if output_csv:
            csv_file.close()

        if output:
            out.release()

        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)

        self.cfd.reset()

        if sound_notifications:
            self.sound_notifications_lock = False


    def notification_thread(self):
        while self.sound_notifications_lock:
            if len(self.notification_queue) > 0:
                class_id = self.notification_queue.pop()

                notification_sound = AudioSegment.from_mp3('/home/arian/Projects/traffic-signs-detector/data/sounds/notification.mp3')
                play(notification_sound)

                class_name_sound = AudioSegment.from_mp3('/home/arian/Projects/traffic-signs-detector/data/sounds/'+ str(class_id) + '.mp3')
                play(class_name_sound)


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

    detections = detector.detect_video_feed(0, show_output=True, sound_notifications=True)
