from numpy import ndarray

def remove_dirt(
    image: ndarray,
    keep: bool = True,
    max_distance: int = 20,
    min_area: float = 0.05,
) -> ndarray: ...
def fill_holes(
    image: ndarray, fill_value: int, hole_area: float = 0.001
) -> ndarray: ...
def refine_regions(
    image: ndarray, body_labels: set[int], min_area: float = 0.01
) -> ndarray: ...
def refine_legs(
    image: ndarray,
    leg_labels: set[int],
    pair_labels: list[tuple[int, int]],
    body_labels: set[int],
    alternative_labels: set[int] = {},
) -> ndarray: ...
def leg_segments(
    image: ndarray,
    labels: dict[int, list[int]],
    body_labels: set[int],
    alternative_labels: set[int] = {},
) -> ndarray: ...
