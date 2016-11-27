import numpy as np


def get_index_of_max_element(arr):
    """Returns the index of the highest element in the 1D numpy.ndarray.
    Ties are broken randomly.

    Args:
        arr (1D numpy.ndarray): Array of elements.

    Returns:
        int: Index of the highest element.
    """
    secondary = np.random.random(arr.size)
    sort_indices = np.lexsort((secondary, arr))
    return sort_indices[-1]
