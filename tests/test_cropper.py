import cv2
import unittest
import numpy as np

from .context import detector
from detector.cropper import Cropper


class TestCropper:


    def test_expand_coordinates_inside(self):
        crop_percent = 0.1
        force_square = False

        image_size = (1920, 1080)

        x1 = 500
        y1 = 300

        width = 100
        height = 200

        original_coordinates = (x1, y1, x1 + width, y1 + height)

        crp = Cropper(crop_percent, force_square)

        expanded_coordinates =  crp.expand_coordinates(original_coordinates, image_size)
        expected_coordinates = (350, 200, 750, 600)

        assert expanded_coordinates == expected_coordinates


    def test_expand_coordinates_right_side(self):
        crop_percent = 0.1
        force_square = False

        image_size = (600, 800)

        x1 = 500
        y1 = 300

        width = 100
        height = 200

        original_coordinates = (x1, y1, x1 + width, y1 + height)

        crp = Cropper(crop_percent, force_square)

        expanded_coordinates =  crp.expand_coordinates(original_coordinates, image_size)
        expected_coordinates = (350, 200, 600, 600)

        assert expanded_coordinates == expected_coordinates


    def test_expand_coordinates_left_side(self):
        crop_percent = 0.1
        force_square = False

        image_size = (1920, 1080)

        x1 = 0
        y1 = 500

        width = 100
        height = 200

        original_coordinates = (x1, y1, x1 + width, y1 + height)

        crp = Cropper(crop_percent, force_square)

        expanded_coordinates =  crp.expand_coordinates(original_coordinates, image_size)
        expected_coordinates = (0, 400, 250, 800)

        assert expanded_coordinates == expected_coordinates


    def test_expand_coordinates_top_side(self):
        crop_percent = 0.1
        force_square = False

        image_size = (1920, 1080)

        x1 = 500
        y1 = 0

        width = 200
        height = 100

        original_coordinates = (x1, y1, x1 + width, y1 + height)

        crp = Cropper(crop_percent, force_square)

        expanded_coordinates =  crp.expand_coordinates(original_coordinates, image_size)
        expected_coordinates = (400, 0, 800, 250)

        assert expanded_coordinates == expected_coordinates


    def test_expand_coordinates_bottom_side(self):
        crop_percent = 0.1
        force_square = False

        image_size = (1920, 1080)

        x1 = 500
        y1 = 1000

        width = 100
        height = 80

        original_coordinates = (x1, y1, x1 + width, y1 + height)

        crp = Cropper(crop_percent, force_square)

        expanded_coordinates =  crp.expand_coordinates(original_coordinates, image_size)
        expected_coordinates = (450, 940, 650, 1080)

        assert expanded_coordinates == expected_coordinates


    def test_expand_coordinates_square_inside(self):
        crop_percent = 0.1
        force_square = True

        image_size = (1920, 1080)

        x1 = 500
        y1 = 300

        width = 100
        height = 200

        original_coordinates = (x1, y1, x1 + width, y1 + height)

        crp = Cropper(crop_percent, force_square)

        expanded_coordinates =  crp.expand_coordinates(original_coordinates, image_size)
        expected_coordinates = (440, 290, 660, 510)

        assert expanded_coordinates == expected_coordinates


    def test_expand_coordinates_square_right_side(self):
        crop_percent = 0.1
        force_square = True

        image_size = (600, 800)

        x1 = 500
        y1 = 300

        width = 100
        height = 200

        original_coordinates = (x1, y1, x1 + width, y1 + height)

        crp = Cropper(crop_percent, force_square)

        expanded_coordinates =  crp.expand_coordinates(original_coordinates, image_size)
        expected_coordinates = (440, 290, 600, 510)

        assert expanded_coordinates == expected_coordinates


    def test_expand_coordinates_square_left_side(self):
        crop_percent = 0.1
        force_square = True

        image_size = (1920, 1080)

        x1 = 0
        y1 = 500

        width = 100
        height = 200

        original_coordinates = (x1, y1, x1 + width, y1 + height)

        crp = Cropper(crop_percent, force_square)

        expanded_coordinates =  crp.expand_coordinates(original_coordinates, image_size)
        expected_coordinates = (0, 490, 160, 710)

        assert expanded_coordinates == expected_coordinates


    def test_expand_coordinates_square_top_side(self):
        crop_percent = 0.1
        force_square = True

        image_size = (1920, 1080)

        x1 = 500
        y1 = 0

        width = 200
        height = 100

        original_coordinates = (x1, y1, x1 + width, y1 + height)

        crp = Cropper(crop_percent, force_square)

        expanded_coordinates =  crp.expand_coordinates(original_coordinates, image_size)
        expected_coordinates = (490, 0, 710, 160)

        assert expanded_coordinates == expected_coordinates


    def test_expand_coordinates_square_bottom_side(self):
        crop_percent = 0.1
        force_square = True

        image_size = (1920, 1080)

        x1 = 500
        y1 = 1000

        width = 100
        height = 80

        original_coordinates = (x1, y1, x1 + width, y1 + height)

        crp = Cropper(crop_percent, force_square)

        expanded_coordinates =  crp.expand_coordinates(original_coordinates, image_size)
        expected_coordinates = (495, 985, 605, 1080)

        assert expanded_coordinates == expected_coordinates


    def test_expand_and_crop(self):
        crop_percent = 0.1
        force_square = True

        test_image_path = 'tests/data/test.jpg'
        image = cv2.imread(test_image_path)

        h, w, _ = image.shape

        x1 = 100
        y1 = 50

        width = 200
        height = 300

        original_coordinates = (x1, y1, x1 + width, y1 + height)

        crp = Cropper(crop_percent, force_square)

        crop_image =  crp.expand_and_crop(image, original_coordinates)

        crop_h, crop_w, _ = crop_image.shape

        assert crop_h == 330
        assert crop_w == 330
