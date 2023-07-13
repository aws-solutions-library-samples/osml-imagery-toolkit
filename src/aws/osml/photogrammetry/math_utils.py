from math import sqrt
from typing import List


def equilateral_triangle(centroid: List[float], size: float) -> List[List[float]]:
    """
    Utility function to create an equilateral triangle with a given side length centered on a point. There
    are many solutions to that problem so this returns the solution where the top corner is directly above
    the center and the base are horizontal. (i.e. for a triangle made of points ABC A[0] == center[0] and
    B[1] == C[1])

    :param centroid: the center of the triangle
    :param size: the length of one side

    :return: a list of lists representing the triangle len(result) == 3, len(result[0]) == 2
    """
    sized_triangle_at_origin = [
        [0.0, size * sqrt(3.0) / 3.0],
        [-1.0 * size / 2.0, -1.0 * sqrt(3) / 6.0 * size],
        [size / 2.0, -1.0 * sqrt(3) / 6.0 * size],
    ]

    return [[centroid[0] + coord[0], centroid[1] + coord[1]] for coord in sized_triangle_at_origin]
