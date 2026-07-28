from unittest import TestCase

from pandas import DataFrame
from image_info import ImageInfo
class TestImageInfo(TestCase):
    def setUp(self):
        self.imageInfo = ImageInfo("street.jpg")
    def test_boxes_class(self):
        self.assertEqual([2, 3, 4, 5, 6], self.imageInfo.boxes_class("person"))
        self.assertEqual([7, 9], self.imageInfo.boxes_class("suitcase") + self.imageInfo.boxes_class("handbag"))
        self.assertEqual([0, 8], self.imageInfo.boxes_class("car"))
    def test_box_info(self):
        tuple_exp = (
            1434.1,	1108.2,	2802.4,	2321.8,	0.9, "car"	
        )
        tuple_act = self.imageInfo.box_info(0)
        for exp, act in zip(tuple_exp, tuple_act):
            if isinstance(exp, float):
                self.assertAlmostEqual(exp, act, places=1)
            else:
                self.assertEqual(exp, act)    
    def test_data_frame(self) :
        df1: DataFrame = self.imageInfo.data_frame()
        df2: DataFrame = self.imageInfo.data_frame()
        self.assertTrue(df1.equals(df2)) 
        self.assertEqual((10, 6), df1.shape)
        self.assertEqual("car",df1["name"][0])
        self.assertAlmostEqual(0.3, df1["confidence"][9],places=1)
    def test_suitcase_handbag_belongings(self):
        belongings_dict = self.imageInfo.suitcase_handbag_person(0.1)
        self.assertEqual(4, belongings_dict[7][0])
        self.assertEqual(4, belongings_dict[9][0])
        self.assertTrue(belongings_dict[9][1] < belongings_dict[7][1] < 0.1)
    def test_handbag_belongings(self) :
        belongings_dict = self.imageInfo.suitcase_handbag_person(0.08)
        self.assertEqual(4, belongings_dict[9][0])
        self.assertIsNone(belongings_dict[7])
    def test_no_belongings(self) :
        belongings_dict = self.imageInfo.suitcase_handbag_person(0.05)
        self.assertIsNone(belongings_dict[7])
        self.assertIsNone(belongings_dict[9])
          