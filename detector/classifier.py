import cv2
import keras
import numpy as np

from detector.exceptions import ValueOutOfBoundsException

class Classifier:

    def __init__(self, model_path, weights_path, labels_path, threshold, skip_classes=[]):
        with open(model_path, 'r') as json_file:
            model_json = json_file.read()

        self.__classifier_model = keras.models.model_from_json(model_json)
        self.__classifier_model.load_weights(weights_path)

        self.__threshold = threshold

        self.__skip_classes = skip_classes

        labels = open(labels_path, 'r').read().strip().split('\n')

        self.__class_map = {}
        class_id = 0

        for label in labels:
            self.__class_map[class_id] = label
            class_id += 1

        self.__input_size = tuple(self.__classifier_model.layers[0].get_output_at(0).get_shape().as_list()[1:-1])


    def classify_image(self, image):
        # Resize image
        image = image[..., ::-1]
        image = cv2.resize(image, self.__input_size)

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        # Predict class for image
        predictions = self.__classifier_model.predict(image, verbose=False)[0]
        class_id = np.argmax(predictions)

        prediction = None

        # If the prediction has greater confidence than the threshold
        if predictions[class_id] > self.__threshold and class_id not in self.__skip_classes:
            # Populate the prediction object
            prediction = {
                'class_id': int(class_id),
                'label': self.__class_map[class_id],
                'confidence': float(predictions[class_id]),
            }

        return prediction


    def set_threshold(self, threshold):
        if threshold >= 0.0 and threshold  <= 1.0:
            self.__threshold = threshold
        else:
            raise ValueOutOfBoundsException('Threshold must be a value between 0 and 1')


    def get_threshold(self):
        return self.__threshold
