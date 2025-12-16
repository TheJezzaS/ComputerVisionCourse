"""Stereo matching."""
import numpy as np
from scipy.signal import convolve2d


class Solution:
    def __init__(self):
        pass

    @staticmethod
    def ssd_distance(left_image: np.ndarray,
                     right_image: np.ndarray,
                     win_size: int,
                     dsp_range: int) -> np.ndarray:
        """Compute the SSDD distances tensor.

        Args:
            left_image: Left image of shape: HxWx3, and type np.double64.
            right_image: Right image of shape: HxWx3, and type np.double64.
            win_size: Window size odd integer.
            dsp_range: Half of the disparity range. The actual range is
            -dsp_range, -dsp_range + 1, ..., 0, 1, ..., dsp_range.

        Returns:
            A tensor of the sum of squared differences for every pixel in a
            window of size win_size X win_size, for the 2*dsp_range + 1
            possible disparity values. The tensor shape should be:
            HxWx(2*dsp_range+1).
        """
        num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
        disparity_values = list(range(-dsp_range, dsp_range+1))
        ssdd_tensor = np.zeros((num_of_rows,
                                num_of_cols,
                                len(disparity_values)))

        half = win_size // 2

        for i in range(num_of_rows):
            for j in range(num_of_cols):
                for d_idx, disparity in enumerate(disparity_values):
                    for r in range(-half, half + 1):
                        for c in range(-half, half + 1):

                            li = i + r
                            lj = j + c
                            rj = j + c + disparity

                            # Left pixel valid?
                            if 0 <= li < num_of_rows and 0 <= lj < num_of_cols:
                                left_val = left_image[li, lj]
                            else:
                                left_val = 0

                            # Right pixel valid?
                            if 0 <= li < num_of_rows and 0 <= rj < num_of_cols:
                                right_val = right_image[li, rj]
                            else:
                                right_val = 0

                            ssdd_tensor[i, j, d_idx] += np.sum((left_val - right_val) ** 2)

        ssdd_tensor -= ssdd_tensor.min()
        ssdd_tensor /= ssdd_tensor.max()
        ssdd_tensor *= 255.0
        return ssdd_tensor

    @staticmethod
    def naive_labeling(ssdd_tensor: np.ndarray) -> np.ndarray:
        """Estimate a naive depth estimation from the SSDD tensor.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.

        Evaluate the labels in a naive approach. Each value in the
        result tensor should contain the disparity matching minimal ssd (sum of
        squared difference).

        Returns:
            Naive labels HxW matrix.
        """
        # you can erase the label_no_smooth initialization.
        label_no_smooth = np.zeros((ssdd_tensor.shape[0], ssdd_tensor.shape[1]))

        label_no_smooth = np.argmin(ssdd_tensor, axis=2)

        return label_no_smooth

    @staticmethod
    def dp_grade_slice(c_slice: np.ndarray, p1: float, p2: float) -> np.ndarray:
        """Calculate the scores matrix for slice c_slice.

        Calculate the scores slice which for each column and disparity value
        states the score of the best route. The scores slice is of shape:
        (2*dsp_range + 1)xW.

        Args:
            c_slice: A slice of the ssdd tensor.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Scores slice which for each column and disparity value states the
            score of the best route.
        """
        num_labels, num_of_cols = c_slice.shape[0], c_slice.shape[1]
        l_slice = np.zeros((num_labels, num_of_cols))
        l_slice[:,0] = c_slice[:,0] # init the first col to be first col of c

        for col in range(1, num_of_cols):
            term3 = np.min(l_slice[:,col-1])
            for d in range(num_labels):
                # find C
                C = c_slice[d, col]

                # find M
                m1 = l_slice[d,col-1]

                # find 𝐿(𝑑 − 1, 𝑐𝑜𝑙 − 1), 𝐿(𝑑 + 1, 𝑐𝑜𝑙 − 1) if they exist
                if d - 1 >= 0:
                    l1 = l_slice[d - 1, col - 1]
                else:
                    l1 = np.inf
                if d + 1 < num_labels:
                    l2 = l_slice[d + 1, col - 1]
                else:
                    l2 = np.inf
                m2 = p1 + min(l1, l2)

                l3 = np.inf
                for k in range(num_labels):
                    if k == d-1 or k == d or k == d+1:
                        continue
                    l3_temp = l_slice[k, col-1]
                    if l3_temp < l3:
                        l3 = l3_temp

                m3 = p2 + l3
                M = min(m1, m2, m3)
                l_slice[d, col] = C + M - term3
        return l_slice

    # @staticmethod
    # def dp_grade_pixel(c_slice: np.ndarray, p1: float, p2: float) -> np.ndarray:
    #     pass

    def dp_labeling(self,
                    ssdd_tensor: np.ndarray,
                    p1: float,
                    p2: float) -> np.ndarray:
        """Estimate a depth map using Dynamic Programming.

        (1) Call dp_grade_slice on each row slice of the ssdd tensor.
        (2) Store each slice in a corresponding l tensor (of shape as ssdd).
        (3) Finally, for each pixel in l (along each row and column), choose
        the best disparity value. That is the disparity value which
        corresponds to the lowest l value in that pixel.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Dynamic Programming depth estimation matrix of shape HxW.
        """
        l = np.zeros_like(ssdd_tensor)

        for row in range(ssdd_tensor.shape[0]):
            l[row, :, :] = self.dp_grade_slice(ssdd_tensor[row, :, :].T, p1, p2).T


        return self.naive_labeling(l)

    @staticmethod
    def _get_slices_by_direction(ssdd_tensor: np.ndarray,
                                    direction: int):
        """The function gets slices from a given ssdd tensor according to a given direction between 1 and 4
            Note; the opposite directions can be obtained by simply flipping the output slices.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            direction: An integer direction between 1-4
        Returns:
            A list of tuples in the form of (slice, offset) where offset
            gives the value of the offset depends on the direction:
            - for horzintol OR vertical directions: the row OR col number of the slice
            - for diagonal direction: the offset relative to the main diagonal (main diagonal offset=0),
            negative offset are for diagonals below and positive are for diagonals above.
        """
        h, w, d = ssdd_tensor.shape
        slices = [] # list of tupples to store pairs of (slice, offset of the diagonal relativly to the main diagonal)

        if direction == 1:
            # Left to right (Horizontal)
            slices = [(ssdd_tensor[y, :, :].T, y) for y in range(h)]

        elif direction == 2:
            # Top-Left to Bottom-Right (Diagonal)
            for offset in range(-w + 1, h):
                slices.append((np.diagonal(ssdd_tensor, offset=offset, axis1=1, axis2=0), offset))

        elif direction == 3:
            # Top to bottom (Vertical)
            slices = [(ssdd_tensor[:, x, :].T,x) for x in range(w)]

        elif direction == 4:
            # Top-Right to Bottom-Left (Diagonal)
            for offset in range(-w + 1, h):
                slices.append((np.diagonal(np.flip(ssdd_tensor, axis=1), offset=offset, axis1=1, axis2=0), offset))

        return slices

    @staticmethod
    def _get_pixel_coordinates(direction: int, offset: int, col_idx: int, height: int, width: int):
        """
        Map a slice column index back to the (y,x) pixel coordinates in the original tensor.

        Args:
            direction: int 1-4 for horizontal, diagonal TL-BR, vertical, anti-diagonal
            offset: offset returned by _get_slices_by_direction
            col_idx: index of the column in the slice
            height: image height
            width: image width

        Returns:
            (y, x) tuple for original tensor
        """
        if direction == 1:  # Horizontal
            y = offset
            x = col_idx

        elif direction == 2:  # Diagonal TL-BR
            # Compute start coordinates based on offset
            if offset >= 0:
                y_start = 0
                x_start = offset
            else:
                y_start = -offset
                x_start = 0
            y = y_start + col_idx
            x = x_start + col_idx

        elif direction == 3:  # Vertical
            y = col_idx
            x = offset

        elif direction == 4:  # Anti-diagonal TR-BL
            if offset >= 0:
                y_start = 0
                x_start = width - 1 - offset
            else:
                y_start = -offset
                x_start = width - 1
            y = y_start + col_idx
            x = x_start - col_idx

        else:
            raise ValueError(f"Direction {direction} not supported in this helper")

        # Ensure coordinates are within bounds
        y = min(max(y, 0), height - 1)
        x = min(max(x, 0), width - 1)

        return y, x

    def dp_labeling_per_direction(self,
                                  ssdd_tensor: np.ndarray,
                                  p1: float,
                                  p2: float) -> dict:
        """Return a dictionary of directions to a Dynamic Programming
        etimation of depth.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Dictionary int->np.ndarray which maps each direction to the
            corresponding dynamic programming estimation of depth based on
            that direction.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        direction_to_slice = {}

        # direction 1 + 5 (horizontal directions)
        l_first, l_second = self._dp_score_by_direction(ssdd_tensor, p1, p2, 1)
        direction_to_slice[1], direction_to_slice[5] = np.argmin(l_first, axis=2), np.argmin(l_second, axis=2)

        # directions 3 + 7 (vertical directions)
        l_first, l_second = self._dp_score_by_direction(ssdd_tensor, p1, p2, 3)
        direction_to_slice[3], direction_to_slice[7] = np.argmin(l_first, axis=2), np.argmin(l_second, axis=2)

        # direction 2 + 6
        l_first, l_second = self._dp_score_by_direction(ssdd_tensor, p1, p2, 2)
        direction_to_slice[2], direction_to_slice[6] = np.argmin(l_first, axis=2), np.argmin(l_second, axis=2)

        # directions 4 + 8
        l_first, l_second = self._dp_score_by_direction(ssdd_tensor, p1, p2, 4)
        direction_to_slice[4], direction_to_slice[8] = np.argmin(l_first, axis=2), np.argmin(l_second, axis=2)

        return direction_to_slice

    def _get_slices_by_direction(self,
                                    ssdd_tensor: np.ndarray,
                                    direction: int):
        """The function extracts slices from the given ssdd tensor according to a given direction between 1 and 4
            to get each one of the opposite directions it is enough to flip the output slices.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            direction: An integer specifing a direction between 1-4
        Returns:
            A list of tuples in the form of (slice, offset) where offset 
            specefies the value of the offset depends on the direction oriantation:
            - for horzintol OR vertical directions: the row OR col number of the slice
            - for diagonal direction: the offset relative to the main diagonal (main diagonal offset=0),
            negative offset are for diagonals below and positive are for diagonals above.
        """
        h, w, d = ssdd_tensor.shape
        slices = [] # list of tupples to store pairs of (slice, offset of the diagonal relativly to the main diagonal)

        if direction == 1:
            # Left to right (Horizontal)
            slices = [(ssdd_tensor[y, :, :].T, y) for y in range(h)]

        elif direction == 2:
            # Top-Left to Bottom-Right (Diagonal)
            for offset in range(-w + 1, h):
                slices.append((np.diagonal(ssdd_tensor, offset=offset, axis1=1, axis2=0), offset))

        elif direction == 3:
            # Top to bottom (Vertical)
            slices = [(ssdd_tensor[:, x, :].T,x) for x in range(w)]

        elif direction == 4:
            # Top-Right to Bottom-Left (Diagonal)
            for offset in range(-w + 1, h):
                slices.append((np.diagonal(np.flip(ssdd_tensor, axis=1), offset=offset, axis1=1, axis2=0), offset))

        return slices

    def _dp_score_by_direction(self,
                              ssdd_tensor: np.ndarray,
                              p1: float,
                              p2: float,
                              direction: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns the dynamic programming estimation of the ssdd matrix given,
        by scanning the matrix in a given direction

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
            direction: the direction of the scan

        Returns:
            A Tuple of 2 np.ndarrays of dimensions same as ssdd tensor containing the scoring result
            first array for first direction, second array for the opposite direction.
            (To reduce extracting the diagonals for the opposite directions)
        """
        l_first = np.zeros_like(ssdd_tensor)
        l_second = np.zeros_like(ssdd_tensor)
        (H, W, D) = ssdd_tensor.shape

        if direction == 1 or direction == 5:
            slices = self._get_slices_by_direction(ssdd_tensor, direction=1)
            for slice, offset in slices:
                l_first[offset, :, :] = self.dp_grade_slice(slice, p1, p2).T
                l_second[offset, :, :] = np.flip(self.dp_grade_slice(np.flip(slice, axis=1), p1, p2),
                                                 axis=1).T  # for the other direction we flip the vector

            return (l_first, l_second)

        elif direction == 3 or direction == 7:
            slices = self._get_slices_by_direction(ssdd_tensor, direction=3)
            for slice, offset in slices:
                l_first[:, offset, :] = self.dp_grade_slice(slice, p1, p2).T
                l_second[:, offset, :] = np.flip(self.dp_grade_slice(np.flip(slice, axis=1), p1, p2), axis=1).T

            return (l_first, l_second)

        elif direction == 2 or direction == 6:
            slices = self._get_slices_by_direction(ssdd_tensor, direction=2)
            for slice, offset in slices:
                # diagonal with offset=OFF, coordinates are of the form [x, x - OFF]
                x_start = max(0, offset)
                H_eff = min(H - offset, H)
                W_eff = min(W + offset, W)
                x_end = min(W_eff, H_eff) + x_start
                x_range = range(x_start, x_end)
                y_range = [x - offset for x in x_range]

                l_first[x_range, y_range] = self.dp_grade_slice(slice, p1, p2).T
                l_second[x_range, y_range] = np.flip(self.dp_grade_slice(np.flip(slice, axis=1), p1, p2), axis=1).T

            return (l_first, l_second)

        elif direction == 4 or direction == 8:
            slices = self._get_slices_by_direction(ssdd_tensor, direction=4)

            for slice, offset in slices:
                # diagonal with offset=OFF, coordinates are of the form [x, x - OFF]
                x_start = max(0, offset)
                H_eff = min(H - offset, H)
                W_eff = min(W + offset, W)
                x_end = min(W_eff, H_eff) + x_start
                x_range = range(x_start, x_end)
                y_range = [W - 1 - (x - offset) for x in
                           x_range]  # the indecies are the same but mirrored for the y axis

                l_first[x_range, y_range] = self.dp_grade_slice(slice, p1, p2).T
                l_second[x_range, y_range] = np.flip(self.dp_grade_slice(np.flip(slice, axis=1), p1, p2), axis=1).T

            return (l_first, l_second)

    def sgm_labeling(self, ssdd_tensor: np.ndarray, p1: float, p2: float):
        """Estimate the depth map according to the SGM algorithm.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Semi-Global Mapping depth estimation matrix of shape HxW.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)

        for direction in range(1, int((num_of_directions / 2) + 1)):
            first_l, second_l = self._dp_score_by_direction(ssdd_tensor, p1, p2, direction)
            l += first_l
            l += second_l

        l /= num_of_directions

        return self.naive_labeling(l)
