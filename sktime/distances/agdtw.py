__author__ = "Ansgar Asseburg"
__email__ = "devaa@donnerluetjen.de"

from copy import deepcopy

import numpy as np


def agdtw_distance(first, second, window=1, sigma=1.0):
    """
    idea and algorithm was taken from:

    @inproceedings{XueZTWL17,
      author    = {Yangtao Xue and
                   Li Zhang and
                   Zhiwei Tao and
                   Bangjun Wang and
                   Fanzhang Li},
      editor    = {Derong Liu and
                   Shengli Xie and
                   Yuanqing Li and
                   Dongbin Zhao and
                   El{-}Sayed M. El{-}Alfy},
      title     = {An Altered Kernel Transformation for Time Series
      Classification},
      booktitle = {Neural Information Processing - 24th International
      Conference, {ICONIP}
                   2017, Guangzhou, China, November 14-18, 2017,
                   Proceedings, Part {V}},
      series    = {Lecture Notes in Computer Science},
      volume    = {10638},
      pages     = {455--465},
      publisher = {Springer},
      year      = {2017},
      url       = {https://doi.org/10.1007/978-3-319-70139-4\_46},
      doi       = {10.1007/978-3-319-70139-4\_46},
      timestamp = {Tue, 14 May 2019 10:00:42 +0200},
      biburl    = {https://dblp.org/rec/conf/iconip/XueZTWL17.bib},
      bibsource = {dblp computer science bibliography, https://dblp.org}
    }

    the method accepts two univariate time series, eg. 2D single row arrays
    the passed in time series are stripped from NANs
    @param first: numpy array containing the first time series
    @param second: numpy array containing the second time series
    @param window: float representing the window width as ratio of the window
    and the longer series
    @param sigma: float representing the kernel parameter
    @return: a float containing the kernel distance
    """

    # make sure time series are univariate
    if first.shape[0] * second.shape[0] != 1:
        raise ValueError("time series must be univariate!")

    # reduce series to 1D arrays and replace NANs
    first = replace_nans(first.squeeze())
    second = replace_nans(second.squeeze())
    # make sure first series is not longer than second one
    if len(first) > len(second):
        first, second = second, first
    pairwise_distances = get_pairwise_distances(first, second)
    warp_matrix = warping_matrix(pairwise_distances, window)
    pairwise_similarities = get_pairwise_agdtw_similarities(first, second,
                                                            sigma)
    index_of_last_cell = tuple(np.subtract(warp_matrix.shape, 1))

    warping_path_result = kernel_result(index_of_last_cell,
                                        warp_matrix, pairwise_similarities,
                                        result_store={})
    # normalize the similarity value by dividing by the number of values
    # accumulated
    return warping_path_result['similarity']


def replace_nans(series):
    return np.nan_to_num(series)


def get_pairwise_distances(first, second):
    """
    calculates the pairwise squared euclidean distances for the two series
    @param first: np.array containing the first series
    @param second: np.array containing the second series
    @return: np.array containing a matrix with pairwise euclidean
    distances
    """
    return np.power(np.subtract.outer(first, second) ** 2, 0.5)


def get_pairwise_agdtw_similarities(first, second, sigma):
    """
    calculates the pairwise agdtw similarity values for the two series
    @param first: np.array containing the first series
    @param second: np.array containing the second series
    @param sigma: a number containing the sigma value for the agdtw calculation
    @return: np.array containing a matrix with agdtw similarity values
    """

    return np.exp(- np.power(np.divide(np.subtract.outer(first, second),
                                       sigma), 2))


def warping_matrix(pairwise_distances, window=1.0):
    """
    Creates the warping matrix while respecting a given window
    *** part of the code was adopted from elastic.py by Jason Lines ***
    @param pairwise_distances: numpy array containing the distances between any
    two points in first and second series
    @param window: float representing the window width as ratio of the
    window and the longer series
    @return: 2D numpy array containing the minimum squared distances
    """

    row_dim = pairwise_distances.shape[0]
    col_dim = pairwise_distances.shape[1]

    # 1 >= window >= 0
    window = min(1.0, abs(window))
    # working on indices requires absolute_window_size to be integer
    absolute_window_size = int(max(row_dim, col_dim) * window)
    if row_dim + absolute_window_size < col_dim:
        raise ValueError("window ist too small!")
    # initialise matrix
    warp_matrix = np.full([row_dim, col_dim], np.inf)

    # then initialise edges of the warping matrix with accumulated distances
    warp_matrix[0] = np.cumsum(pairwise_distances[0])
    for row in range(1, warp_matrix.shape[0]):
        warp_matrix[row][0] = warp_matrix[row - 1][0] \
                              + pairwise_distances[row][0]

    # now visit all allowed cells, calculate the value as the distance
    # in this cell + min(top, left, or top-left)
    # traverse each row,
    for row in range(1, row_dim):
        # traverse left and right by the allowed window
        window_range_start = row - absolute_window_size
        window_range_end = row + absolute_window_size + 1
        for column in range(window_range_start, window_range_end):
            if not 1 <= column < col_dim:
                continue

            # find smallest entry in the warping matrix, either above,
            # to the left, or diagonally left and up
            neighbor_minimum = np.amin(warp_matrix[row - 1:row + 1,
                                       column - 1:column + 1])

            # add the pairwise distance for [row][column] to the minimum
            # of the three possible potential cells
            warp_matrix[row][column] = \
                pairwise_distances[row][column] + neighbor_minimum

    return warp_matrix


def indices_of_minimum_neighbors(matrix, current_index=(0, 0),
                                 visited_neighbors={}):
    """ finds indices pointing to minimum values within the neighboring cells
    of current_index
    @param matrix: numpy array with the original warping matrix
    @param current_index: tuple pointing to the current index
    @param visited_neighbors: dictionary containing cells that must not be
    visited anymore
    @return: a list containing all the index tuples that point to the minimum
    value among the neighbors
    """
    northern_row = max(current_index[0] - 1, 0)
    western_col = max(current_index[1] - 1, 0)
    # indices to all neighbors
    above = (northern_row, current_index[1])
    left_above = (northern_row, western_col)
    left = (current_index[0], western_col)
    # remove duplicates
    neighbors = set([above, left_above, left])
    neighbors = [x for x in neighbors if (x not in visited_neighbors)]
    neighbors_minimum = np.amin([matrix[x[0]][x[1]] for x in neighbors])
    minimum_neighbors = [index for index in neighbors
                         if matrix[index[0]][index[1]] == neighbors_minimum]
    return minimum_neighbors


def kernel_result(index, warping_matrix, pairwise_similarities,
                  result_store={}, visited_neighbors={}):
    """
    calculates the kernel distance by processing each individual distance
    along the warping squared_euclidean_distances
    @param index: tuple containing the current index into the warping matrix
    @param warping_matrix: numpy array containing the warping matrix
    @param pairwise_similarities: a numpy array containing the pairwise
    similarity values
    @param result_store: dictionary storing the results for memoization -
    needs to be accessible to all recursion levels, thus must be
    passed by reference
    @param visited_neighbors: dictionary holding the visited neighbors for all
    downstream recursions - it must be passed by value
    @return: dict containing the similarity value and the warping path length
    """
    # return early if result is known already
    if index in result_store:
        return result_store[index]

    if index == (0, 0):  # base condition
        return {'similarity': pairwise_similarities[index[0]][index[1]],
                'wp_length': 1}

    # copy visited_neighbors to emulate pass-by-value
    local_visited_neighbors = deepcopy(visited_neighbors)
    # and add current index
    local_visited_neighbors[index] = True
    # find all neighboring cells containing
    # the minimum value among the neighbors
    all_minimum_neighbor_cell_indexes = indices_of_minimum_neighbors(
        warping_matrix, index, local_visited_neighbors)
    # find the similarity values for all those minimum neighbor cells
    min_neighbor_results = np.array(
        [kernel_result(cell, warping_matrix, pairwise_similarities,
                       result_store, local_visited_neighbors) for cell
         in all_minimum_neighbor_cell_indexes])
    # take the maximum similarity value found
    min_neighbor_similarities = [r['similarity'] for r in min_neighbor_results]
    min_neighbor_wp_lengths = [r['wp_length'] for r in min_neighbor_results]
    max_similarity = np.amax(min_neighbor_similarities)
    wp_length = min_neighbor_wp_lengths[
        min_neighbor_similarities.index(max_similarity)]
    # and add this cell's similarity value
    # since every recursion averages the similarity value we'll undo this here
    similarity_result = max_similarity \
                        + pairwise_similarities[index[0]][index[1]]
    # add this step to warping path length
    wp_length_result = wp_length + 1
    # store result for memoization
    result = {'similarity': similarity_result,
              'wp_length': wp_length_result}
    result_store[index] = result
    # ToDo: divide the result by the length of the warping path; every
    #  recursion result will be divided, thus when you have a result r from
    #  a previous recursion with a pyth length of l, the new result is
    #  (r*l+new_r)/(l+1)
    return result


if __name__ == '__main__':
    import os
    import time
    from sktime.utils.data_io import load_from_arff_to_dataframe
    from sktime.classification.distance_based import \
        KNeighborsTimeSeriesClassifier

    from sklearn.metrics import (
        accuracy_score,
        recall_score,
        f1_score,
    )

    DATA_PATH = "../datasets/data"
    DATASET = "MedicalImages"

    X_train, y_train = load_from_arff_to_dataframe(
        os.path.join(DATA_PATH, f"{DATASET}/{DATASET}_TRAIN.arff")
    )
    X_test, y_test = load_from_arff_to_dataframe(
        os.path.join(DATA_PATH, f"{DATASET}/{DATASET}_TEST.arff")
    )

    knn = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw")
    knn.fit(X_train, y_train)

    start_time = time.perf_counter()

    y_test_pred = knn.predict(X_test)
    print("accuracy: ", accuracy_score(y_test, y_test_pred))
    print("recall: ", recall_score(y_test, y_test_pred, average='macro'))
    print("f1 score: ", f1_score(y_test, y_test_pred, average='macro'))

    end_time = time.perf_counter()
    print(f"Elapsed Time: {(end_time - start_time):.3e} s")
