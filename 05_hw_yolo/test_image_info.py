import math
from unittest import TestCase

from pandas import DataFrame

from image_info import BoxInfo, ImageInfo, _apply_threshold, _distance, _min_distance_index


class TestImageInfo(TestCase):
    """Integration tests: run real YOLO inference on street.jpg.

    device is pinned to "cpu" here so the exact box coordinates/confidence
    checked below stay reproducible on any machine, regardless of which
    device ImageInfo picks by default in production (device=None -> best
    available). The image is inferred once for the whole class since
    ImageInfo is read-only after construction.
    """

    @classmethod
    def setUpClass(cls):
        cls.image_info = ImageInfo("street.jpg", device="cpu")

    def test_boxes_class(self):
        self.assertEqual([2, 3, 4, 5, 6], self.image_info.boxes_class("person"))
        self.assertEqual(
            [7, 9],
            self.image_info.boxes_class("suitcase") + self.image_info.boxes_class("handbag"),
        )
        self.assertEqual([0, 8], self.image_info.boxes_class("car"))

    def test_box_info(self):
        expected = BoxInfo(1434.1, 1108.2, 2802.4, 2321.8, 0.9, "car")
        actual = self.image_info.box_info(0)
        for exp, act in zip(expected, actual):
            if isinstance(exp, float):
                self.assertAlmostEqual(exp, act, places=1)
            else:
                self.assertEqual(exp, act)

    def test_data_frame(self):
        df1: DataFrame = self.image_info.data_frame()
        df2: DataFrame = self.image_info.data_frame()
        self.assertIs(df1, df2)
        self.assertTrue(df1.equals(df2))
        self.assertEqual((10, 6), df1.shape)
        self.assertEqual("car", df1["name"][0])
        self.assertAlmostEqual(0.3, df1["confidence"][9], places=1)

    def test_suitcase_handbag_belongings(self):
        belongings_dict = self.image_info.suitcase_handbag_person(0.1)
        self.assertEqual(4, belongings_dict[7][0])
        self.assertEqual(4, belongings_dict[9][0])
        self.assertTrue(belongings_dict[9][1] < belongings_dict[7][1] < 0.1)

    def test_handbag_belongings(self):
        belongings_dict = self.image_info.suitcase_handbag_person(0.08)
        self.assertEqual(4, belongings_dict[9][0])
        self.assertIsNone(belongings_dict[7])

    def test_no_belongings(self):
        belongings_dict = self.image_info.suitcase_handbag_person(0.05)
        self.assertIsNone(belongings_dict[7])
        self.assertIsNone(belongings_dict[9])


class TestBelongingsMatching(TestCase):
    """Unit tests for the pure matching logic - no model, no image, fully deterministic."""

    def test_distance(self):
        self.assertAlmostEqual(0.0, _distance((0.1, 0.2), (0.1, 0.2)))
        self.assertAlmostEqual(5.0, _distance((0.0, 0.0), (3.0, 4.0)))

    def test_min_distance_index_picks_closest(self):
        centers = {0: (0.1, 0.1), 1: (0.9, 0.9), 2: (0.12, 0.11)}
        result = _min_distance_index(0, [1, 2], centers.get)
        self.assertEqual(2, result[0])
        self.assertAlmostEqual(math.hypot(0.02, 0.01), result[1], places=6)

    def test_min_distance_index_no_candidates(self):
        self.assertIsNone(_min_distance_index(0, [], {0: (0, 0)}.get))

    def test_apply_threshold(self):
        matches = {7: (4, 0.05), 9: (4, 0.2), 11: None}
        result = _apply_threshold(matches, threshold=0.1)
        self.assertEqual((4, 0.05), result[7])
        self.assertIsNone(result[9])
        self.assertIsNone(result[11])
