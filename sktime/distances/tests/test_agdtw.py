__author__ = "Ansgar Asseburg"
__email__ = "devaa@donnerluetjen.de"

import pytest
import sktime.distances.tests._config as cfg

"""
run on commandline from root with:
    ptw --runner "pytest sktime/distances/tests/test_agdtw.py" 
        -- --last-failed --new-first
"""


def pytest_assertrepr_compare(op, left, right):
    import numpy as np
    if op == '==' and (
            isinstance(left, np.array) and isinstance(right, np.array)
    ) or (
            isinstance(left, tuple) and isinstance(right, tuple)
    ):
        return [f'{left} in {right}']


def test_pairwise_distance():
    import numpy as np
    import sktime.distances.agdtw as agdtw

    series_1 = np.array([1, 2, 3, 2, 2])
    series_2 = np.array([5, 7, 4, 4, 3, 2])

    expected_result = [[4, 6, 3, 3, 2, 1],
                       [3, 5, 2, 2, 1, 0],
                       [2, 4, 1, 1, 0, 1],
                       [3, 5, 2, 2, 1, 0],
                       [3, 5, 2, 2, 1, 0]]
    actual_result = agdtw.get_pairwise_distances(series_1, series_2)

    assert (expected_result == actual_result).all()


def test_pairwise_similarities_with_sigma_1_short():
    import numpy as np
    import sktime.distances.agdtw as agdtw

    series_1 = np.array([1, 2])
    series_2 = np.array([5, 7, 4])

    expected_result = [
        [1.12535174719259e-07,
         2.31952283024357e-16,
         1.2340980408668e-04],
        [1.2340980408668e-04,
         1.3887943864964e-11,
         1.83156388887342e-02]]

    actual_result = agdtw.get_pairwise_agdtw_similarities(series_1,
                                                          series_2, 1)
    for i in range(len(expected_result)):
        assert (pytest.approx(expected_result[i], 0.001) == actual_result[i])


def test_pairwise_similarities_with_sigma_1_and_first_series_long():
    import numpy as np
    import sktime.distances.agdtw as agdtw

    series_1 = np.array([5, 7, 4, 4, 3, 2])
    series_2 = np.array([1, 2, 3, 2, 2])

    expected_result = [
        [1.12535174719259e-07, 1.2340980408668e-04, 1.83156388887342e-02,
         1.2340980408668e-04, 1.2340980408668e-04],
        [2.31952283024357e-16, 1.3887943864964e-11, 1.12535174719259e-07,
         1.3887943864964e-11, 1.3887943864964e-11],
        [1.2340980408668e-04, 1.83156388887342e-02, 3.67879441171442e-01,
         1.83156388887342e-02, 1.83156388887342e-02],
        [1.2340980408668e-04, 1.83156388887342e-02, 3.67879441171442e-01,
         1.83156388887342e-02, 1.83156388887342e-02],
        [1.83156388887342e-02, 3.67879441171442e-01, 1e+00,
         3.67879441171442e-01, 3.67879441171442e-01],
        [3.67879441171442e-01, 1e+00, 3.67879441171442e-01, 1e+00, 1e+00]]

    actual_result = agdtw.get_pairwise_agdtw_similarities(series_1,
                                                          series_2, 1)
    for i in range(len(expected_result)):
        assert (pytest.approx(expected_result[i], 0.001) == actual_result[i])


def test_pairwise_similarities_with_sigma_1_and_first_series_short():
    import numpy as np
    import sktime.distances.agdtw as agdtw

    series_1 = np.array([1, 2, 3, 2, 2])
    series_2 = np.array([5, 7, 4, 4, 3, 2])

    expected_result = [
        [1.12535174719259e-07, 2.31952283024357e-16, 1.2340980408668e-04,
         1.2340980408668e-04, 1.83156388887342e-02, 3.67879441171442e-01],
        [1.2340980408668e-04, 1.3887943864964e-11, 1.83156388887342e-02,
         1.83156388887342e-02, 3.67879441171442e-01, 1e+00],
        [1.83156388887342e-02, 1.12535174719259e-07, 3.67879441171442e-01,
         3.67879441171442e-01, 1e+00, 3.67879441171442e-01],
        [1.2340980408668e-04, 1.3887943864964e-11, 1.83156388887342e-02,
         1.83156388887342e-02, 3.67879441171442e-01, 1e+00],
        [1.2340980408668e-04, 1.3887943864964e-11, 1.83156388887342e-02,
         1.83156388887342e-02, 3.67879441171442e-01, 1e+00]]

    actual_result = agdtw.get_pairwise_agdtw_similarities(series_1,
                                                          series_2, 1)
    for i in range(len(expected_result)):
        assert (pytest.approx(expected_result[i], 0.001) == actual_result[i])


def test_pairwise_similarities_with_sigma_one_half():
    import numpy as np
    import sktime.distances.agdtw as agdtw

    series_1 = np.array([1, 2])
    series_2 = np.array([5, 7, 4])

    expected_result = [
        [1.60381089054864e-28,
         2.8946403116483e-63,
         2.31952283024357e-16],
        [2.31952283024357e-16,
         3.72007597602084e-44,
         1.12535174719259e-07]]

    actual_result = agdtw.get_pairwise_agdtw_similarities(series_1,
                                                          series_2, 0.5)
    for i in range(len(expected_result)):
        assert (pytest.approx(expected_result[i], 0.00001) == actual_result[i])


def test_indices_of_minimum_neighbors():
    import numpy as np
    import sktime.distances.agdtw as agdtw

    test_matrix = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [1, 7, 0]
    ])

    test_range = np.array([
        [(0, 0), (0, 1)],
        [(0, 1), (0, 2)],
        [(0, 0), (1, 0)],
        [(0, 0), (1, 1)],
        [(0, 1), (1, 2)],
        [(1, 0), (2, 0)],
        [(2, 0), (2, 1)],
        [(1, 1), (2, 2)]
    ])

    for expected_indices, source_indices in test_range:
        source_index = tuple(source_indices)
        actual_indices = \
            agdtw.indices_of_minimum_neighbors(test_matrix, source_index,
                                               {source_index})
        assert (expected_indices == actual_indices).all(), \
            f'at source indices: {tuple(source_indices)}'


def test_warping_matrix_with_window_1():
    import numpy as np
    import sktime.distances.agdtw as agdtw

    series_1 = np.array([1, 2, 3, 2, 2])
    series_2 = np.array([5, 7, 4, 4, 3, 2])
    pairwise_distances = agdtw.get_pairwise_distances(series_1, series_2)
    expected_result = np.array([[4, 10, 13, 16, 18, 19],
                                [7, 9, 11, 13, 14, 14],
                                [9, 11, 10, 11, 11, 12],
                                [12, 14, 12, 12, 12, 11],
                                [15, 17, 14, 14, 13, 11]])
    actual_result = agdtw.warping_matrix(pairwise_distances)

    assert actual_result.shape == expected_result.shape
    assert (actual_result == expected_result).all()


def test_warping_matrix_with_window_a_quarter():
    import numpy as np
    import sktime.distances.agdtw as agdtw

    series_1 = np.array([1, 2, 3, 4, 5])
    series_2 = np.array([1, 2, 3, 4, 5])
    pairwise_distances = agdtw.get_pairwise_distances(series_1, series_2)
    expected_result = np.array([[0, 1, 3, 6, 10],
                                [1, 0, 1, np.inf, np.inf],
                                [3, 1, 0, 1, np.inf],
                                [6, np.inf, 1, 0, 1],
                                [10, np.inf, np.inf, 1, 0]])
    actual_result = agdtw.warping_matrix(pairwise_distances, window=.25)

    assert actual_result.shape == expected_result.shape
    assert (actual_result == expected_result).all()


def test_warping_matrix_with_window_as_zero():
    import numpy as np
    import sktime.distances.agdtw as agdtw

    series_1 = np.array([1, 2, 3, 4, 5])
    series_2 = np.array([1, 2, 3, 4, 5])
    pairwise_distances = agdtw.get_pairwise_distances(series_1, series_2)
    expected_result = np.array([[0, 1, 3, 6, 10],
                                [1, 0, np.inf, np.inf, np.inf],
                                [3, np.inf, 0, np.inf, np.inf],
                                [6, np.inf, np.inf, 0, np.inf],
                                [10, np.inf, np.inf, np.inf, 0]])
    actual_result = agdtw.warping_matrix(pairwise_distances, window=0)

    assert actual_result.shape == expected_result.shape
    assert (actual_result == expected_result).all()


def test_kernel_result_changes_visited_only_locally():
    import numpy as np
    import sktime.distances.agdtw as agdtw
    series_1 = np.array([1, 2, 3, 2, 2])
    series_2 = np.array([5, 7, 4, 4, 3, 2])
    pairwise_distances = agdtw.get_pairwise_distances(series_1, series_2)
    pairwais_similarities = agdtw.get_pairwise_agdtw_similarities(series_1,
                                                                  series_2, 1)
    # warping_matrix = warp_matrix = np.zeros([5, 5])
    warping_matrix = agdtw.warping_matrix(pairwise_distances)
    visited = {}
    expected_result = 0
    index = tuple(np.subtract(warping_matrix.shape, 1))
    agdtw.kernel_result(index, warping_matrix,
                        pairwais_similarities, {}, visited)
    actual_result = len(visited)
    assert (expected_result == actual_result)


def test_kernel_result_changes_result_store_globally():
    import numpy as np
    import sktime.distances.agdtw as agdtw
    series_1 = np.array([1, 2, 3, 2, 2])
    series_2 = np.array([5, 7, 4, 4, 3, 2])
    pairwise_distances = agdtw.get_pairwise_distances(series_1, series_2)
    pairwais_similarities = agdtw.get_pairwise_agdtw_similarities(series_1,
                                                                  series_2, 1)
    # warping_matrix = warp_matrix = np.zeros([5, 5])
    warping_matrix = agdtw.warping_matrix(pairwise_distances)
    visited = {}
    expected_minimum_result = 6
    result_store = {}
    index = tuple(np.subtract(warping_matrix.shape, 1))
    agdtw.kernel_result(index, warping_matrix,
                        pairwais_similarities, result_store, visited)
    actual_result = len(result_store)
    assert (expected_minimum_result <= actual_result)


def test_kernel_result_result_store_is_used():
    import numpy as np
    import sktime.distances.agdtw as agdtw
    from unittest.mock import Mock

    # we need to restore the function for subsequent tests
    store = agdtw.indices_of_minimum_neighbors
    agdtw.indices_of_minimum_neighbors = Mock()
    series_1 = np.array([1, 2, 3, 2, 2])
    series_2 = np.array([5, 7, 4, 4, 3, 2])
    pairwise_distances = agdtw.get_pairwise_distances(series_1, series_2)
    pairwais_similarities = agdtw.get_pairwise_agdtw_similarities(series_1,
                                                                  series_2, 1)
    # warping_matrix = warp_matrix = np.zeros([5, 5])
    warping_matrix = agdtw.warping_matrix(pairwise_distances)
    visited = {}
    result_store = {(4, 5): 42}
    index = tuple(np.subtract(warping_matrix.shape, 1))
    agdtw.kernel_result(index, warping_matrix,
                        pairwais_similarities, result_store, visited)

    agdtw.indices_of_minimum_neighbors.assert_not_called()
    agdtw.indices_of_minimum_neighbors = store


@pytest.mark.parametrize("series_1, series_2, expected_result",
                         cfg.KERNEL_TEST_SAMPLE)
def test_kernel_result_returns_dict_of_similarity_and_length(series_1,
                                                             series_2,
                                                             expected_result):
    import numpy as np
    import sktime.distances.agdtw as agdtw
    pairwise_distances = agdtw.get_pairwise_distances(series_2, series_1)
    pairwise_similarities = agdtw.get_pairwise_agdtw_similarities(series_2,
                                                                  series_1, 1)
    # warping_matrix = warp_matrix = np.zeros([5, 5])
    warping_matrix = agdtw.warping_matrix(pairwise_distances)
    visited = {}
    result_store = {}
    index = tuple(np.subtract(warping_matrix.shape, 1))
    actual_result = agdtw.kernel_result(index, warping_matrix,
                                        pairwise_similarities, result_store,
                                        visited)
    assert actual_result['similarity'] == pytest.approx(
        expected_result['similarity'], 1e-5)
    assert actual_result['wp_length'] == expected_result['wp_length']


@pytest.mark.parametrize("series_1, series_2", cfg.MULTIVARIATES)
def test_agdtw_distance_throws_for_multivariates(series_1, series_2):
    import sktime.distances.agdtw as agdtw
    with pytest.raises(ValueError) as e_info:
        agdtw.agdtw_distance(series_1, series_2)
    assert "univariate" in str(e_info.value)


@pytest.mark.parametrize("series_1, series_2", cfg.UNIVARIATES)
def test_agdtw_distance_returns_single_value(series_1, series_2):
    from numbers import Number
    import sktime.distances.agdtw as agdtw

    actual_result = agdtw.agdtw_distance(series_1, series_2)
    assert isinstance(actual_result, Number)


@pytest.mark.parametrize("series_1, series_2, expected_result",
                         cfg.AGDTW_SAMPLE)
def test_agdtw_distance_returns_correct_result(series_1, series_2,
                                               expected_result):
    import sktime.distances.agdtw as agdtw
    actual_result = agdtw.agdtw_distance(series_1, series_2)
    assert actual_result == pytest.approx(expected_result, 0.00001)


@pytest.mark.parametrize("series_1, series_2", cfg.NAN_SAMPLES)
def test_series_are_stripped_from_NaNs(series_1, series_2):
    import sktime.distances.agdtw as agdtw
    import numpy as np

    sum_not_NaN = np.sum(agdtw.strip_nans(series_1))
    assert not np.isnan(sum_not_NaN)
    sum_not_NaN = np.sum(agdtw.strip_nans(series_2))
    assert not np.isnan(sum_not_NaN)
