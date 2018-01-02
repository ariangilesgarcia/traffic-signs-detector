import json

from detector.utils import convert_detection_to_csv
from detector.utils import convert_detection_to_yolo


class Logger:

    def save_detections_to_csv(self, detections, output):
        with open(output, 'w') as fp:
            for detection in detections:
                csv_detection = convert_detection_to_csv(detection)
                fp.write(csv_detection + '\n')


    def save_detections_to_json(self, detections, output):
        with open(output, 'w') as fp:
            json.dump(detections, fp)


    def save_detections_to_yolo(self, detections, image_size, output):
        with open(output, 'w') as fp:
            for detection in detections:
                yolo_detection = convert_detection_to_yolo(detection, image_size)
                fp.write(yolo_detection + '\n')
