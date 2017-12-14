import numpy as np


class CrossFrameDetector:

    def __init__(self, num_classes, frame_history_count, frames_threshold):
        # Number of object classes for the CFD to detect
        self.num_classes = num_classes

        # Objects must be correctly detected in <frames_threshold>
        # out of the <frame_history_count> latest frames in order
        # to be deteceted by the CrossFrameDetector
        self.frame_history_count = frame_history_count
        self.frames_threshold = frames_threshold

        # Matrix of size <frame_history_count> x <num_classes>
        # Each row represents detections of a frame
        # Each column represents an object class
        self.last_detections = np.zeros(shape=(self.frame_history_count, self.num_classes))

        # Detection mask
        self.detection_mask = np.zeros(shape=(self.num_classes))

        # Count, decrements 1 each frame
        self.detection_mask_limit = 100


    def register_detections(self, detections):
        """
        Update detection counter.
        Given a list with the detected signs, it updates the detections
        counter used for filtering out false positives.
        Args:
            detected_signs_list (list): array of detections.
        Returns:
            detections (list): list of actually detected classes id.
        """

        for detection in detections:
            detected_signs_list


        self.last_detections = np.roll(self.last_detections, -1, axis=0)
        self.last_detections[self.frame_count - 1, :] = 0
        self.last_detections[self.frame_count - 1, detected_signs_list] = 1

        frames_sum = np.sum(self.last_detections, axis=0)
        detected_classes_ids = np.where(frames_sum > self.frame_detection_thresh)

        for detected_class in detected_classes_ids[0]:
            if self.detection_mask[detected_class] == 0:
                self.detection_mask[detected_class] = 1

        self.detection_mask[self.detection_mask > 0] += 1

        self.detection_mask[self.detection_mask > self.detection_mask_limit] = 0

        detections = self.get_detections()

        return detections

"""
    def get_detections(self):
        \"""
        Get actually detected signs.
        Returns:
            detected_classes_ids (list): list of actually detected classes id.
        \"""

        detected_classes_ids = np.where(self.detection_mask > 0)
        return detected_classes_ids
"""
