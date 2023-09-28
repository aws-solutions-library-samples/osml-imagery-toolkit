from typing import List, Optional


class DRAParameters:
    """
    This class manages a set of parameters used to perform a Dynamic Range Adjustment that is applied when
    converting imagery pixel values (e.g. 11-bit per pixel panchromatic imagery to an 8-bit per pixel grayscale).
    """

    def __init__(
        self, suggested_min_value: float, suggested_max_value: float, actual_min_value: float, actual_max_value: float
    ):
        """
        Constructor for this class.

        :param suggested_min_value: suggested minimum value of the relevant pixel range
        :param suggested_max_value: suggested maximum value of the relevant pixel range
        :param actual_min_value: actual minimum value of pixels in the image
        :param actual_max_value: actual maximum value of pixels in the image
        """
        self.suggested_min_value = suggested_min_value
        self.suggested_max_value = suggested_max_value
        self.actual_min_value = actual_min_value
        self.actual_max_value = actual_max_value

    @staticmethod
    def from_counts(
        counts: List[float],
        first_bucket_value: Optional[float] = None,
        last_bucket_value: Optional[float] = None,
        min_percentage: float = 0.02,
        max_percentage: float = 0.98,
        a: float = 0.2,
        b: float = 0.4,
    ) -> "DRAParameters":
        """
        This static factory method computes a new set of DRA parameters given a histogram of pixel values.

        :param counts: histogram of the pixel values
        :param first_bucket_value: pixel value of the first bucket, defaults to 0
        :param last_bucket_value: pixel value of the last bucket, defaults to bucket index
        :param min_percentage: set point for low intensity pixels that may be outliers
        :param max_percentage: set point for high intensity pixels that may be outliers
        :param a: weighting factor for the low intensity range
        :param b: weighting factor for the high intensity range
        :return: a set of DRA parameters containing recommended and actual ranges of values
        """
        num_histogram_bins = len(counts)
        if not first_bucket_value:
            first_bucket_value = 0
        if not last_bucket_value:
            last_bucket_value = num_histogram_bins

        # Find the first and last non-zero counts
        actual_min_value = 0
        while actual_min_value < num_histogram_bins and counts[actual_min_value] == 0:
            actual_min_value += 1

        actual_max_value = num_histogram_bins - 1
        while actual_max_value > 0 and counts[actual_max_value] == 0:
            actual_max_value -= 1

        # Compute the cumulative distribution
        cumulative_counts = counts.copy()
        for i in range(1, len(cumulative_counts)):
            cumulative_counts[i] = cumulative_counts[i] + cumulative_counts[i - 1]

        # Find the values that exclude the lowest and highest percentages of the counts.
        # This identifies the range that contains most of the pixels while excluding outliers.
        max_counts = cumulative_counts[-1]
        low_threshold = min_percentage * max_counts
        e_min = 0
        while cumulative_counts[e_min] < low_threshold:
            e_min += 1

        high_threshold = max_percentage * max_counts
        e_max = num_histogram_bins - 1
        while cumulative_counts[e_max] > high_threshold:
            e_max -= 1

        min_value = max([actual_min_value, e_min - a * (e_max - e_min)])
        max_value = min([actual_max_value, e_max + b * (e_max - e_min)])

        value_step = (last_bucket_value - first_bucket_value) / num_histogram_bins
        return DRAParameters(
            suggested_min_value=min_value * value_step + first_bucket_value,
            suggested_max_value=max_value * value_step + first_bucket_value,
            actual_min_value=actual_min_value * value_step + first_bucket_value,
            actual_max_value=actual_max_value * value_step + first_bucket_value,
        )

    def __repr__(self):
        return (
            f"DRAParameters(min_value={self.suggested_min_value}, "
            f"max_value={self.suggested_max_value}, "
            f"e_first={self.actual_min_value}, "
            f"e_last={self.actual_max_value}, "
            f")"
        )
