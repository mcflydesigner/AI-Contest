import unittest
import cv2
from numpy.testing import assert_allclose, assert_equal
import numpy as np

from main import ExtendedImage, GeneticAlgorithm

PATH_TO_TEST_IMG = './images/test.jpg'


class TestGeneticMethod(unittest.TestCase):
    def setUp(self):
        # Working with BGR
        self.img = cv2.imread(PATH_TO_TEST_IMG)
        self.ex_img = ExtendedImage(self.img)

    def test_proper_right_test_image(self):
        default_err = 'Not required input image'
        self.assertEqual(self.ex_img.get_width(), 16, default_err)
        self.assertEqual(self.ex_img.get_height(), 16, default_err)
        self.assertEqual(self.img[0, 0].tolist(), [1, 63, 255], default_err)
        self.assertEqual(self.img[0, 9].tolist(), [40, 40, 200], default_err)
        self.assertEqual(self.img[9, 0].tolist(), [168, 108, 78], default_err)
        self.assertEqual(self.img[9, 9].tolist(), [0, 0, 0], default_err)

    def test_image_class_avg_color(self):
        assert_allclose(self.ex_img.get_color_in_region((0, 0), (7, 7)), [1, 63, 255])
        assert_allclose(self.ex_img.get_color_in_region((0, 8), (7, 15)), [40, 40, 200])
        assert_allclose(self.ex_img.get_color_in_region((8, 0), (15, 7)), [168, 108, 78])
        assert_allclose(self.ex_img.get_color_in_region((8, 8), (15, 15)), [0, 0, 0])

        assert_allclose(self.ex_img.get_color_in_region((7, 7), (8, 8)), [52.25, 52.75, 133.25])

    def test_image_class_sizes(self):
        self.assertEqual(self.ex_img.get_width(), 16)
        self.assertEqual(self.ex_img.get_height(), 16)

        blank_ext_img = ExtendedImage.create_empty_image(width=10,
                                                         height=20)
        self.assertEqual(blank_ext_img.get_width(), 10)
        self.assertEqual(blank_ext_img.get_height(), 20)

        blank_ext_img = ExtendedImage.create_empty_image(width=20,
                                                         height=30)
        self.assertEqual(blank_ext_img.get_width(), 20)
        self.assertEqual(blank_ext_img.get_height(), 30)

    def test_genetic_algorithm_class_init(self):
        with self.assertRaises(AssertionError):
            GeneticAlgorithm(self.img, 3, 10, 10)

    def test_genetic_algorithm_class_сrossing_over(self):
        blank_ext_img = ExtendedImage.create_empty_image(width=4,
                                                         height=4)
        img_ext_cropped = ExtendedImage(self.img[0:4, 0:4])
        ga = GeneticAlgorithm(self.img, 10, 10, 10)

        expected_result = np.array([
            [[1, 63, 255], [1, 63, 255], [0, 0, 0], [0, 0, 0]],
            [[1, 63, 255], [1, 63, 255], [0, 0, 0], [0, 0, 0]],
            [[1, 63, 255], [1, 63, 255], [0, 0, 0], [0, 0, 0]],
            [[1, 63, 255], [1, 63, 255], [0, 0, 0], [0, 0, 0]]
        ])
        assert_equal(ga._crossing_over(img_ext_cropped, blank_ext_img).img, expected_result)

    def test_genetic_algorithm_class_fit_test(self):
        default_err = 'Incorrect result of `_fit_test` method.'
        blank_ext_img = ExtendedImage.create_empty_image(16, 16)
        # original img as input one

        res = 0
        for i in range(blank_ext_img.get_width()):
            for j in range(blank_ext_img.get_height()):
                r, g, b = (blank_ext_img.img[i, j] - self.img[i, j]) * (blank_ext_img.img[i, j] - self.img[i, j])
                res += sum([r, g, b])

        ga = GeneticAlgorithm(self.img, 10, 10, 10)
        self.assertEqual(ga._fit_test(blank_ext_img), res, default_err)
        self.assertEqual(ga._fit_test(self.ex_img), 0, default_err)


if __name__ == '__main__':
    unittest.main()
