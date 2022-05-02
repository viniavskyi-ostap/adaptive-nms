# distutils: language = c++

import numpy as np
from libc.math cimport ceil, floor, round, sqrt
from libcpp.vector cimport vector

cpdef square_covering_adaptive_nms(keypoints: np.ndarray, responses: np.ndarray,
                                   width: int, height: int, target_num_kpts: int, indices_only: bool = False,
                                   up_tol: int = 10, max_num_iter: int = 30):
    """
    Square covering Adaptive Non-Maximum suppression of 2D keypoints
    Args:
        keypoints: 2D array of keypoints (N, 2)
        responses: 1D array of corresponding kpts responses (N,)
        width: width of image in px where kpts were detected
        height: height of image in px where kpts were detected 
        target_num_kpts: target number of keypoint
        indices_only: return only indices of selected keypoints
        up_tol: tolerance for detection of more keypoints than required
        max_num_iter: maximum number of bianry search iterations
    Returns:
        selected_keypoints: 2D array of keypoints selected after NMS (target_num_kpts, 2)
    """
    # check input parameters
    if len(keypoints.shape) != 2 and keypoints.shape[1] == 2:
        raise ValueError(f'`keypoints` must be a 2-dim array of shape (*, 2). Provided shape: {keypoints.shape}')
    if len(responses.shape) != 1:
        raise ValueError(f'`responses` must be a 1-dim array. Provided number of dim: {len(responses.shape)}')
    if responses.shape[0] != keypoints.shape[0]:
        raise ValueError(f'`keypoints` and `responses` must have equal length along 0-th dimension.'
                         f' Provided lengths: keypoints={keypoints.shape[0]}', responses={responses.shape[0]})
    if width <= 0 or height <= 0:
        raise ValueError('`width` and `height` must be positive integers')
    if target_num_kpts <= 0 or target_num_kpts >= keypoints.shape[0]:
        raise ValueError('`target_num_keypoints` must lie in the interval [1, keypoints.shape[0]-1]')
    if keypoints.dtype != np.float32:
        keypoints = keypoints.astype(np.float32)
    if responses.dtype != np.float32:
        responses = responses.astype(np.float32)
    if up_tol < 0:
        raise ValueError('`up_tol` must be a non-negative integer')
    if max_num_iter <= 0:
        raise ValueError('`max_num_iter` must be a positive integer')

    result = _square_covering_adaptive_nms(keypoints, responses, width, height, target_num_kpts,
                                           indices_only=indices_only, up_tol=up_tol, max_num_iter=max_num_iter)
    return np.asarray(result)

cpdef _square_covering_adaptive_nms(const float[:, :] keypoints, const float[:] responses,
                                    Py_ssize_t width, Py_ssize_t height, Py_ssize_t target_num_kpts, bint indices_only,
                                    Py_ssize_t up_tol, unsigned int max_num_iter):
    cdef:
        double low, high, mid
        Py_ssize_t current_num_kpts, i
        unsigned char complete = False
        unsigned int num_iters = 0
        vector[Py_ssize_t] result_kpts_idx

    low = floor(sqrt(keypoints.shape[0] / target_num_kpts))
    high = _adaptive_nms_upper_bound(width, height, target_num_kpts)

    # define keypoints priority
    cdef float[:] neg_responses = np.empty_like(responses, dtype=np.float32)
    # negate responses for further sorting
    for i in range(responses.shape[0]):
        neg_responses[i] = -responses[i]

    cdef Py_ssize_t[:] priority_idxs = np.argsort(neg_responses)

    while not complete:
        mid = (low + high) / 2
        result_kpts_idx = _square_covering_nms(keypoints, priority_idxs, width, height, window_radius=mid)

        current_num_kpts = <Py_ssize_t> result_kpts_idx.size()
        num_iters += 1

        if (target_num_kpts <= current_num_kpts < target_num_kpts + up_tol) or num_iters == max_num_iter:
            complete = True

        if current_num_kpts > target_num_kpts:
            low = mid
        elif current_num_kpts < target_num_kpts:
            high = mid

    cdef long long[:] selected_keypoints_idxs
    if indices_only:
        selected_keypoints_idxs = np.empty((result_kpts_idx.size(),), dtype=np.int64)
        for i in range(<Py_ssize_t> result_kpts_idx.size()):
            selected_keypoints_idxs[i] = result_kpts_idx[i]
        return result_kpts_idx

    cdef float[:, :] selected_keypoints = np.empty((result_kpts_idx.size(), 2), dtype=np.float32)
    for i in range(<Py_ssize_t> result_kpts_idx.size()):
        selected_keypoints[i] = keypoints[result_kpts_idx[i]]
    return selected_keypoints


cdef vector[Py_ssize_t] _square_covering_nms(const float[:, :] keypoints, Py_ssize_t[:] priority_idxs,
                                             size_t width, size_t height, double window_radius):
    """
    Square covering Non-Maximum suppression of 2D keypoints
    Args:
        keypoints: 2D array of keypoints (N, 2)
        priority_idxs: 1D array of keypoints indexes in prioritizing order
        width: width of image in px where kpts were detected
        height: height of image in px where kpts were detected
        window_radius: radius of square kernel in px used for NMS

    Returns:
        selected_keypoints_idx: 1D array of keypoints indices selected after NMS (K,)
    """
    cdef:
        Py_ssize_t i
        double grid_res = window_radius / 2
        Py_ssize_t num_cell_cols = <Py_ssize_t> ceil(width / grid_res)
        Py_ssize_t num_cell_rows = <Py_ssize_t> ceil(height / grid_res)
        unsigned char [:, :] covered_grid = np.zeros((num_cell_rows, num_cell_cols), dtype=np.uint8)
        vector[Py_ssize_t] result_kpts_idx
        Py_ssize_t idx, row, col, row_min, row_max, col_min, col_max

    for i in range(keypoints.shape[0]):
        idx = priority_idxs[i]
        # get position of the cell current point is located at
        row = <Py_ssize_t> floor(keypoints[idx, 1] / grid_res)
        col = <Py_ssize_t> floor(keypoints[idx, 0] / grid_res)

        if not covered_grid[row, col]:  # if the cell is not covered
            result_kpts_idx.push_back(idx)
            # get range which window radius is covering (+- 2 grid cells)
            row_min = max(row - 2, 0)
            row_max = min(row + 2, num_cell_rows - 1)

            col_min = max(col - 2, 0)
            col_max = min(col + 2, num_cell_cols - 1)

            covered_grid[row_min: row_max + 1, col_min: col_max + 1] = 1

    return result_kpts_idx


cdef double _adaptive_nms_upper_bound(Py_ssize_t width, Py_ssize_t height, Py_ssize_t target_num_kpts) nogil:
    """
    Get upper bound on window radius for given image shape and desired number of keypoints
    """
    cdef double exp1, exp2, exp3, exp4, sol

    exp1 = height + width + 2 * target_num_kpts
    exp2 = (
            4 * width
            + 4 * target_num_kpts
            + 4 * height * target_num_kpts
            + height * height
            + width * width
            - 2 * height * width
            + 4 * height * width * target_num_kpts
    )
    exp3 = sqrt(exp2)
    exp4 = target_num_kpts - 1

    sol = -round((exp1 - exp3) / exp4)  # second solution

    return sol
