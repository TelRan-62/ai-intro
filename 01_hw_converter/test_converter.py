from unittest import TestCase

from pandas import DataFrame

from converter import enumerator


class TestConverter(TestCase):
    def setUp(self):
        self.df = DataFrame({
            "Company": ["Toyota", "Toyota", "Hyundai", "Hyundai", "Hyundai"],
            "Model": ["Camry", "Corolla", "i10", "Elantra", "Kona"],
        })

    def test_enumerator(self):
        expected = {
            "Toyota": 0,
            "Hyundai": 1,
        }
        actual = enumerator(self.df["Company"].unique())
        self.assertEqual(expected, actual)