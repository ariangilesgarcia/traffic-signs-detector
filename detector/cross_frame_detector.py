import numpy as np


class CrossFrameDetector:

    def __init__(self, num_classes, frame_history_count, frames_threshold):
        # Number of object classes for the CFD to detect
        self.__num_classes = num_classes

        # Objects must be in <frames_threshold>
        self.__frames_threshold = frames_threshold
        # out of the <frame_history_count> latest frames
        self.__frame_history_count = frame_history_count

        # Rows represents detections in a frame, columns represents object classes
        self.__detection_history= np.zeros(shape=(self.__frame_history_count, self.__num_classes))

        # Detection mask keeps track of detected object classes
        self.__detection_mask = np.zeros(shape=(self.__num_classes))

        # Determins how 'long' the sign will be detected for
        self.__detection_mask_limit = 100


    def register_detections(self, detections):
        # List of ids of detected classes
        detected_object_classes = []

        # Populate list with the detections
        for detection in detections:
            detected_object_classes.append(detection['class_id'])

        # Move everything one row up and complete the last row with the current detections
        self.__detection_history= np.roll(self.__detection_history, -1, axis=0)
        self.__detection_history[self.__frame_history_count - 1, :] = 0
        self.__detection_history[self.__frame_history_count - 1, detected_object_classes] = 1

        # Sum detections over all rows and get de CFD detections
        frames_sum = np.sum(self.__detection_history, axis=0)
        detected_classes_ids = np.where(frames_sum >= self.__frames_threshold)

        # If class was not detected, make the mask 1
        for detected_class in detected_classes_ids[0]:
            if self.__detection_mask[detected_class] == 0:
                self.__detection_mask[detected_class] = 1

        # If class was already detected just add one to the mask
        self.__detection_mask[self.__detection_mask > 0] += 1

        # If class already pass the mask limit, make it 0
        self.__detection_mask[self.__detection_mask > self.__detection_mask_limit] = 0


    def get_detections(self):
        # Get ids of the classes detected by the CFD
        all_detected_classes_ids = np.where(self.__detection_mask > 0)[0].tolist()
        current_detected_classes_ids =  np.where(self.__detection_mask == 2)[0].tolist()

        return all_detected_classes_ids, current_detected_classes_ids


    def get_detection_history_matrix(self):
        return self.__detection_history


    def reset(self):
        self.__detection_history= np.zeros(shape=(self.__frame_history_count, self.__num_classes))
        self.__detection_mask = np.zeros(shape=(self.__num_classes))
