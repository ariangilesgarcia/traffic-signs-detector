import cv2


class Cropper:

    def __init__(self, crop_percent, force_square=False):
        self.__crop_percent = crop_percent
        self.__force_square = force_square


    def expand_coordinates(self, original_coordinates, image_size):
        # Get separate values
        x1, y1, x2, y2 = original_coordinates
        img_width, img_height = image_size

        # Calculate width and height of crop
        width = x2 - x1
        height = y2 - y1

        # Calculate center point of crop
        center_x = x1 + int(width/2)
        center_y = y1 + int(height/2)

        # Calculate the size of the square side
        size = max(width, height)

        # Force square if parameter is set
        if self.__force_square:
            size = int(size/2 * (1 + self.__crop_percent))

        # Calculate new coords
        x1 = center_x - size
        x2 = center_x + size
        y1 = center_y - size
        y2 = center_y + size

        if x1 < 0:
            x1 = 0
        if x2 > img_width:
            x2 = img_width
        if y1 < 0:
            y1 = 0
        if y2 > img_height:
            y2 = img_height

        new_coords = x1, y1, x2, y2

        return new_coords


    def expand_and_crop(self, image, coordinates):
        # Get separate values
        x1, y1, x2, y2 = coordinates

        # Get image size
        h, w, _ = image.shape

        # Expand coordinates
        x1, y1, x2, y2 = self.expand_coordinates(coordinates, (w, h))
        new_coords = [x1, y1, x2, y2]

        # Crop image
        image = image[y1:y2, x1:x2]

        return new_coords, image
