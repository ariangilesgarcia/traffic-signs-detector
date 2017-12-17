

class DetectionPipeline:

    def __init__(self, localizer, cropper, classifier):
        self.localizer = localizer
        self.cropper = cropper
        self.classifier = classifier


    def detect_objects_in_image(self, image):
        object_locations = self.localizer.find_objects_in_image(image)

        classified_detections = []

        for location in object_locations:
            coordinates = location['coordinates']
            new_coords, cropped_image = self.cropper.expand_and_crop(image, coordinates)

            classified_object = self.classifier.classify_image(cropped_image)

            if classified_object:
                classified_object['coordinates'] = new_coords
                classified_detections.append(classified_object)

        return classified_detections
