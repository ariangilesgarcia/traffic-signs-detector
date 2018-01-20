
class DetectionPipeline:

    def __init__(self, localizer, cropper, classifier):
        self.__localizer = localizer
        self.__cropper = cropper
        self.__classifier = classifier


    def detect_objects_in_image(self, image):
        object_locations = self.__localizer.find_objects_in_image(image)

        classified_detections = []

        for location in object_locations:
            coordinates = location['coordinates']
            new_coords, cropped_image = self.__cropper.expand_and_crop(image, coordinates)

            classified_object = self.__classifier.classify_image(cropped_image)

            if classified_object:
                classified_object['coordinates'] = new_coords
                classified_detections.append(classified_object)

        return classified_detections
