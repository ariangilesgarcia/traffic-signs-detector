from darkflow.net.build import TFNet


class Localizer:

    def __init__(self, cfg_path, weights_path, threshold, gpu=0.8):
        options = {
            "model": cfg_path,
            "load": weights_path,
            "threshold": threshold,
            "summary": None,
            "gpu": gpu,
        }

        self.__localizer_model = TFNet(options)


    def find_objects_in_image(self, image):
        detections = self.__localizer_model.return_predict(image)

        new_detections = []

        for detection in detections:
            new_detection = {}

            x1 = detection['topleft']['x']
            y1 = detection['topleft']['y']
            x2 = detection['bottomright']['x']
            y2 = detection['bottomright']['y']

            new_detection['coordinates'] = [x1, y1, x2, y2]
            new_detection['confidence'] = detection['confidence']
            new_detection['label'] = detection['label']
            new_detection['class_id'] = 0


            new_detections.append(new_detection)

        return new_detections
