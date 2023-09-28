import unittest


class TestDRAParameters(unittest.TestCase):
    def test_from_counts(self):
        from aws.osml.gdal.dynamic_range_adjustment import DRAParameters

        counts = [0] * 1024
        counts[1:99] = [1] * (99 - 1)
        counts[100:400] = [200] * (400 - 100)
        counts[1022] = 1

        dra_parameters = DRAParameters.from_counts(counts=counts)

        self.assertEquals(dra_parameters.actual_min_value, 1)
        self.assertEquals(dra_parameters.actual_max_value, 1022)
        self.assertAlmostEqual(dra_parameters.suggested_min_value, 47, delta=1)
        self.assertAlmostEquals(dra_parameters.suggested_max_value, 506, delta=1)
