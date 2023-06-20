from __future__ import annotations

import numpy as np
import numpy.typing as npt


class ProjectiveTransform:
    """
    This is a simple standalone projective transform class with an implementation that only depends on NumPy. There
    are equivalent classes in Open CV and Scikit Imaging but this class can be used when we don't want to include those
    dependencies.
    """

    def __init__(self, matrix_parameters: npt.ArrayLike) -> None:
        """
        Construct a projective transform from the given matrix parameters. Normally this constructor is not called
        directly. See the ProjectiveTransform.estimate() method instead.

        :param matrix_parameters: the matrix parameters for this transformation

        :return: None
        """
        self.matrix_parameters = matrix_parameters

    def forward(self, src_coords: npt.ArrayLike):
        """
        Compute the forward src -> dst transformation for an array of source coordinates [[x, y], ...]

        :param src_coords: the source coordinates

        :return: the array of transformed coordinates [[x',y'], ...]
        """
        a0, a1, a2, b0, b1, b2, c0, c1 = self.matrix_parameters
        x = src_coords[:, 0]
        y = src_coords[:, 1]
        out = np.zeros(src_coords.shape)
        out[:, 0] = (a0 + a1 * x + a2 * y) / (1 + c0 * x + c1 * y)
        out[:, 1] = (b0 + b1 * x + b2 * y) / (1 + c0 * x + c1 * y)
        return out

    def inverse(self, dst_coords: npt.ArrayLike):
        """
        Compute the inverse, dst -> src, transformation for an array of destination coordinates [[x', y'], ...]

        :param dst_coords: the destination coordinates

        :return: the array of transformed coordinates [[x,y], ...]
        """
        a0, a1, a2, b0, b1, b2, c0, c1 = self.matrix_parameters
        x = dst_coords[:, 0]
        y = dst_coords[:, 1]
        out = np.zeros(dst_coords.shape)
        out[:, 0] = (a2 * b0 - a0 * b2 + (b2 - b0 * c1) * x + (a0 * c1 - a2) * y) / (
            a1 * b2 - a2 * b1 + (b1 * c1 - b2 * c0) * x + (a2 * c0 - a1 * c1) * y
        )
        out[:, 1] = (a0 * b1 - a1 * b0 + (b0 * c0 - b1) * x + (a1 - a0 * c0) * y) / (
            a1 * b2 - a2 * b1 + (b1 * c1 - b2 * c0) * x + (a2 * c0 - a1 * c1) * y
        )
        return out

    @classmethod
    def estimate(cls, src: npt.ArrayLike, dst: npt.ArrayLike) -> ProjectiveTransform:
        """
        This method takes a list of source and destination points [x, y] and then applies the least squares fit
        to estimate the projective transform matrix relating them. It then creates the ProjectiveTransform
        object using that matrices. Each list needs to contain at least 4 points.

        :param src: the source points
        :param dst: the destination points

        :return: the projective transform
        """
        xs = src[:, 0]
        ys = src[:, 1]
        num_points = src.shape[0]

        # The Coefficient Matrix
        a = np.zeros((num_points * 2, 8))
        a[:num_points, 0] = 1
        a[:num_points, 1] = xs
        a[:num_points, 2] = ys
        a[num_points:, 3] = 1
        a[num_points:, 4] = xs
        a[num_points:, 5] = ys
        a[:num_points, 6] = -dst[:, 0] * xs
        a[:num_points, 7] = -dst[:, 0] * ys
        a[num_points:, 6] = -dst[:, 1] * xs
        a[num_points:, 7] = -dst[:, 1] * ys

        # The dependent variable values
        b = np.zeros((num_points * 2,))
        b[:num_points] = dst[:, 0]
        b[num_points:] = dst[:, 1]

        # See: https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
        matrix_parameters = np.linalg.lstsq(a, b, rcond=None)[0]

        return cls(matrix_parameters)
