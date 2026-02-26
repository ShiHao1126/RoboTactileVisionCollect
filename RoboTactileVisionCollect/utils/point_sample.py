import math
import random
from typing import List, Tuple

# Small epsilon for float comparisons
_EPS = 1e-9


def generate_sampling_points(
    grid_step: float = 2.5,           # mm
    square_half_width: float = 20.0,   # mm, half side of the bounding square
    radius_limit: float = 18.0,        # mm, points must lie inside this circle
) -> List[Tuple[float, float]]:
    """
    Generate sampling points in a grid centered at (0, 0), then restrict to
    the rectangle [-square_half_width, square_half_width] x [-square_half_width, square_half_width],
    then to the circle of radius radius_limit. Return a random permutation in mm.

    All coordinates are in mm; the caller applies any origin/offset externally.

    Returns
    -------
    List[Tuple[float, float]]
        List of (x_mm, y_mm) in random order.
    """
    points: List[Tuple[float, float]] = []
    max_index = int(round(square_half_width / grid_step))

    # 1. Grid over the square
    for j in range(-max_index, max_index + 1):
        y = j * grid_step
        if abs(y) > square_half_width + _EPS:
            continue
        for i in range(-max_index, max_index + 1):
            x = i * grid_step
            if abs(x) > square_half_width + _EPS:
                continue
            # 2. Restrict to circle
            if math.hypot(x, y) <= radius_limit + _EPS:
                points.append((x, y))

    # 3. Random order
    random.shuffle(points)
    return points   # in mm


if __name__ == "__main__":
    points = generate_sampling_points()
    print(points)
    print(len(points))