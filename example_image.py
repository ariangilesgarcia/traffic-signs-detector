import cv2
from detector.detector import create_detector_from_file

detector = create_detector_from_file('./cfg/example.config.json')
detections = detector.detect_image(cv2.imread('./data/test/test.jpg'), output='predictions.jpg')
print('Detections: ', detections)