"""Projective Homography and Panorama Solution."""
import numpy as np

from typing import Tuple
import random
from collections import namedtuple

import scipy.interpolate
from numpy.linalg import svd
from scipy.interpolate import griddata


PadStruct = namedtuple('PadStruct',
                       ['pad_up', 'pad_down', 'pad_right', 'pad_left'])


class Solution:
    """Implement Projective Homography and Panorama Solution."""
    def __init__(self):
        pass

    @staticmethod
    def compute_homography_naive(match_p_src: np.ndarray,
                                 match_p_dst: np.ndarray) -> np.ndarray:
        """Compute a Homography in the Naive approach, using SVD decomposition.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.

        Returns:
            Homography from source to destination, 3x3 numpy array.
        """
        N = match_p_src.shape[1]


        # step 1, create A
        A = np.zeros([2*N, 9])
        for i in range(N):
            p = match_p_src[:,i]
            p = np.append(p, 1)
            u_prime = match_p_dst[0, i]
            v_prime = match_p_dst[1, i]

            row1 = np.concatenate([
                -p,  # vector
                [0, 0, 0],  # three zeros
                u_prime * p  # vector
            ])
            row2 = np.concatenate([
                [0, 0, 0],  # three zeros
                -p,  # vector
                v_prime * p  # vector
            ])

            A[2*i] = row1
            A[2*i+1] = row2



        # step 2, get eigenvecs of (A^T)A
        eigen_vals, eigen_vecs = np.linalg.eig(np.dot(A.T, A))
        min_index = np.argmin(eigen_vals)
        H = eigen_vecs[:, min_index].reshape(3,3)

        # normalise
        H = H / H[-1, -1]

        return H


    @staticmethod
    def compute_forward_homography_slow(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in the Naive approach, using loops.

        Iterate over the rows and columns of the source image, and compute
        the corresponding point in the destination image using the
        projective homography. Place each pixel value from the source image
        to its corresponding location in the destination image.
        Don't forget to round the pixel locations computed using the
        homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        # #step 1: do Hxp for all p in source image
        Hdst, Wdst, C = dst_image_shape
        src_rows, src_cols = src_image.shape[0], src_image.shape[1]

        new_image = np.zeros((Hdst, Wdst, C), dtype=src_image.dtype)
        for v in range(src_rows):
            for u in range(src_cols):
                p = np.append([u, v], 1)
                p_prime = np.dot(homography, p)
                p_prime /= p_prime[2]  # normalize homogeneous coordinate; we need (u', v', 1)
                new_xidx = int(round(p_prime[0]))
                new_yidx = int(round(p_prime[1]))

                if (0 <= new_xidx < Wdst) and (0 <= new_yidx < Hdst):
                    new_image[new_yidx, new_xidx] = src_image[v, u] # note python inputs y first, then x, smh...
                else:
                    continue

        # import matplotlib.pyplot as plt
        # plt.imshow(new_image)
        return new_image



    @staticmethod
    def compute_forward_homography_fast(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in a fast approach, WITHOUT loops.

        (1) Create a meshgrid of columns and rows.
        (2) Generate a matrix of size 3x(H*W) which stores the pixel locations
        in homogeneous coordinates.
        (3) Transform the source homogeneous coordinates to the target
        homogeneous coordinates with a simple matrix multiplication and
        apply the normalization you've seen in class.
        (4) Convert the coordinates into integer values and clip them
        according to the destination image size.
        (5) Plant the pixels from the source image to the target image according
        to the coordinates you found.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination.
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        Hdst, Wdst, C = dst_image_shape
        src_rows, src_cols = src_image.shape[0], src_image.shape[1]

        # step 1; make a meshgrid of columns  u and rows v for the source image
        u_vals = np.arange(src_cols)
        v_vals = np.arange(src_rows)
        U, V = np.meshgrid(u_vals, v_vals)  # shapes (rows, cols)

        # step 2; homogeneous coordinates 3 x (H*W)
        ones = np.ones_like(U).ravel() # ravel makes it 1D
        pts_src = np.vstack([U.ravel(), V.ravel(), ones])  # 3 x N_src

        # step 3; transform to target homogeneous coordinates
        pts_dst = np.dot(homography, pts_src)  # 3 x N_src
        normalizer = pts_dst[2, :]
        pts_dst = pts_dst[:2, :] / normalizer  # 2 x N_src

        # step4; integer coordinates and clip to destination size (u' in [0,Wdst-1], v' in [0,Hdst-1])
        u_dst = np.round(pts_dst[0, :]).astype(int)
        v_dst = np.round(pts_dst[1, :]).astype(int)

        valid_mask = (u_dst >= 0) & (u_dst < Wdst) & (v_dst >= 0) & (v_dst < Hdst)

        # step 5; plant pixels
        new_image = np.zeros((Hdst, Wdst, C), dtype=src_image.dtype)
        src_flat = src_image.reshape(-1, C)
        valid_src_indices = np.nonzero(valid_mask)[0]
        u_dst_valid = u_dst[valid_mask]
        v_dst_valid = v_dst[valid_mask]

        # transfer pixels
        new_image[v_dst_valid, u_dst_valid, :] = src_flat[valid_src_indices, :]
        # import matplotlib.pyplot as plt
        # plt.imshow(new_image)
        return new_image


    @staticmethod
    def test_homography(homography: np.ndarray,
                        match_p_src: np.ndarray,
                        match_p_dst: np.ndarray,
                        max_err: float) -> Tuple[float, float]:
        """Calculate the quality of the projective transformation model.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.

        Returns:
            A tuple containing the following metrics to quantify the
            homography performance:
            fit_percent: The probability (between 0 and 1) validly mapped src
            points (inliers).
            dist_mse: Mean square error of the distances between validly
            mapped src points, to their corresponding dst points (only for
            inliers). In edge case where the number of inliers is zero,
            return dist_mse = 10 ** 9.
        """
        N = match_p_src.shape[1]

        # step 1; map all source points using the homography
        ones = np.ones((1, N))
        pts_src = np.vstack([match_p_src, ones])  # 3 x N
        pts_dst = np.dot(homography, pts_src)
        normalizer = pts_dst[2, :]
        mapped = pts_dst[:2, :] / normalizer  # 2 x N

        # step 2; calc euclidean distances
        diffs = mapped - match_p_dst
        dists = np.sqrt(np.sum(diffs ** 2, axis=0))
        inliers_mask = (dists <= max_err)
        inliers_count = int(np.sum(inliers_mask))
        fit_percent = inliers_count / N

        if inliers_count == 0:
            dist_mse = 10 ** 9
        else:
            dist_mse = float(np.mean((dists[inliers_mask]) ** 2))

        # dist_mse = float(np.mean((dists[inliers_mask]) ** 2))

        return fit_percent, dist_mse

    @staticmethod
    def points_are_collinear(pts: np.ndarray, tol: float = 1e-6) -> bool:
        # pts: 2x4 array
        x = pts[0, :]
        y = pts[1, :]
        # Compute area of triangle for 3 points at a time
        for i in range(4):
            j = (i + 1) % 4
            k = (i + 2) % 4
            area = 0.5 * np.abs(
                x[i] * (y[j] - y[k]) + x[j] * (y[k] - y[i]) + x[k] * (y[i] - y[j])
            )
            if area < tol:
                return True
        return False

    @staticmethod
    def meet_the_model_points(homography: np.ndarray,
                              match_p_src: np.ndarray,
                              match_p_dst: np.ndarray,
                              max_err: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return which matching points meet the homography.

        Loop through the matching points, and return the matching points from
        both images that are inliers for the given homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            A tuple containing two numpy nd-arrays, containing the matching
            points which meet the model (the homography). The first entry in
            the tuple is the matching points from the source image. That is a
            nd-array of size 2xD (D=the number of points which meet the model).
            The second entry is the matching points form the destination
            image (shape 2xD; D as above).
        """
        N = match_p_src.shape[1]

        ones = np.ones((1, N))
        pts_src = np.vstack([match_p_src, ones])  # 3 x N
        pts_dst = np.dot(homography, pts_src)
        normalizer = pts_dst[2, :]
        mapped = pts_dst[:2, :] / normalizer  # 2 x N

        diffs = mapped - match_p_dst
        dists = np.sqrt(np.sum(diffs ** 2, axis=0))
        inliers_idx = np.where(dists <= max_err)[0]

        p_src_meets_model = match_p_src[:, inliers_idx]
        p_dst_meets_model = match_p_dst[:, inliers_idx]

        return p_src_meets_model, p_dst_meets_model


    def compute_homography(self,
                           match_p_src: np.ndarray,
                           match_p_dst: np.ndarray,
                           inliers_percent: float,
                           max_err: float) -> np.ndarray:
        """Compute homography coefficients using RANSAC to overcome outliers.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            homography: Projective transformation matrix from src to dst.
        """
        # RANSAC parameters
        p = 0.99  # desired confidence
        w = float(inliers_percent)
        n = 4  # minimal sample size for homography
        d = 0.5  # minimal fraction of points that meet the model for early stopping

        # compute number of iterations
        k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** n))) + 1
        print('k', k)

        N = match_p_src.shape[1]
        idxs = list(range(N))

        best_H = None
        best_error = float('inf')
        best_inliers_count = -1
        best_inliers = None

        # random.seed(2)

        for _ in range(k):
            # Step 1: randomly pick n points
            smpl = random.sample(idxs, n)
            src_sample = match_p_src[:, smpl]
            dst_sample = match_p_dst[:, smpl]

            try:
                # Step 2: compute homography from minimal set
                H = self.compute_homography_naive(src_sample, dst_sample)
            except Exception:
                continue

            # Step 3: find inliers from all points
            p_src_all, p_dst_all = self.meet_the_model_points(H, match_p_src, match_p_dst, max_err)
            inliers_count = p_src_all.shape[1]

            # step 4: check inlier fraction
            fit_percent = inliers_count / N
            if fit_percent >= d:
                # Step 4a: if enough inliers, refit model
                if inliers_count >= n:
                    H_inliers = self.compute_homography_naive(p_src_all, p_dst_all)

                    # compute total error over inliers
                    pts_transformed = np.dot(H_inliers, np.vstack([p_src_all, np.ones((1, inliers_count))]))
                    pts_transformed /= pts_transformed[2, :]
                    error = np.sum(np.linalg.norm(pts_transformed[:2, :] - p_dst_all, axis=0))

                    # step 4b: update best model, if improved error
                    if error < best_error:
                        best_error = error
                        best_H = H_inliers
                        best_inliers = (p_src_all, p_dst_all)

        return best_H

    @staticmethod
    def compute_backward_mapping(
            backward_projective_homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute backward mapping.

        (1) Create a mesh-grid of columns and rows of the destination image.
        (2) Create a set of homogenous coordinates for the destination image
        using the mesh-grid from (1).
        (3) Compute the corresponding coordinates in the source image using
        the backward projective homography.
        (4) Create the mesh-grid of source image coordinates.
        (5) For each color channel (RGB): Use scipy's interpolation.griddata
        with an appropriate configuration to compute the bi-cubic
        interpolation of the projected coordinates.

        Args:
            backward_projective_homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination shape.

        Returns:
            The source image backward warped to the destination coordinates.
        """

        # define useful values
        H_dst, W_dst, C = dst_image_shape
        H_src = src_image.shape[0]
        W_src = src_image.shape[1]

        # step 1; make a meshgrid for the destination image
        x_vals = np.arange(W_dst)  # columns
        y_vals = np.arange(H_dst)  # rows
        X_dst, Y_dst = np.meshgrid(x_vals, y_vals)  # shape: xy (cols, rows)

        # step 2; homogeneous coordinates for destination image 3 x (H_dst*W_dst)
        pts_dst = np.vstack([X_dst.ravel(), Y_dst.ravel(), np.ones(H_dst*W_dst)])

        # step 3; apply backwards-homography
        pts_src = backward_projective_homography @ pts_dst  # 3 x N_dst
        pts_src /= pts_src[2, :]  # normalize
        pts_src = pts_src[:2, :]  # keep only pixel locations 2 x N_dst

        x_src = np.round(pts_src[0]).astype(int)
        y_src = np.round(pts_src[1]).astype(int)
        # remove pixels that are out of source-image's bounds:
        valid_mask = ((x_src >= 0) & (x_src < W_src) & (y_src >= 0) & (y_src < H_src))
        pts_src = pts_src[:, valid_mask]
        # the area in destination image that correspopnds to source image bounds
        pts_dst_xy = pts_dst[:2, valid_mask].astype(int)

        # step 4; create meshgrid for the source image
        x_vals = np.arange(W_src)
        y_vals = np.arange(H_src)
        X_src, Y_src = np.meshgrid(x_vals, y_vals)

        points_src = np.stack((X_src.ravel(), Y_src.ravel()), axis=1)  # make the source image points grid

        # step 5; make the backward mapping warp image
        backward_warp = np.zeros(dst_image_shape, dtype=np.uint8)

        for c in range(C):  # for each color channel
            pixel_values = src_image[Y_src.ravel(), X_src.ravel(), c]
            interpolated_data = griddata(points_src, pixel_values, (pts_src[0], pts_src[1]), method='cubic')
            interpolated_data = np.nan_to_num(interpolated_data, nan=0)
            interpolated_data = interpolated_data.astype(int)
            backward_warp[pts_dst_xy[1], pts_dst_xy[0], c] = interpolated_data

        return backward_warp
        pass

    @staticmethod
    def find_panorama_shape(src_image: np.ndarray,
                            dst_image: np.ndarray,
                            homography: np.ndarray
                            ) -> Tuple[int, int, PadStruct]:
        """Compute the panorama shape and the padding in each axes.

        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            homography: 3x3 Projective Homography matrix.

        For each image we define a struct containing it's corners.
        For the source image we compute the projective transformation of the
        coordinates. If some of the transformed image corners yield negative
        indices - the resulting panorama should be padded with at least
        this absolute amount of pixels.
        The panorama's shape should be:
        dst shape + |the largest negative index in the transformed src index|.

        Returns:
            The panorama shape and a struct holding the padding in each axes (
            row, col).
            panorama_rows_num: The number of rows in the panorama of src to dst.
            panorama_cols_num: The number of columns in the panorama of src to
            dst.
            padStruct = a struct with the padding measures along each axes
            (row,col).
        """
        src_rows_num, src_cols_num, _ = src_image.shape
        dst_rows_num, dst_cols_num, _ = dst_image.shape
        src_edges = {}
        src_edges['upper left corner'] = np.array([1, 1, 1])
        src_edges['upper right corner'] = np.array([src_cols_num, 1, 1])
        src_edges['lower left corner'] = np.array([1, src_rows_num, 1])
        src_edges['lower right corner'] = \
            np.array([src_cols_num, src_rows_num, 1])
        transformed_edges = {}
        for corner_name, corner_location in src_edges.items():
            transformed_edges[corner_name] = homography @ corner_location
            transformed_edges[corner_name] /= transformed_edges[corner_name][-1]
        pad_up = pad_down = pad_right = pad_left = 0
        for corner_name, corner_location in transformed_edges.items():
            if corner_location[1] < 1:
                # pad up
                pad_up = max([pad_up, abs(corner_location[1])])
            if corner_location[0] > dst_cols_num:
                # pad right
                pad_right = max([pad_right,
                                 corner_location[0] - dst_cols_num])
            if corner_location[0] < 1:
                # pad left
                pad_left = max([pad_left, abs(corner_location[0])])
            if corner_location[1] > dst_rows_num:
                # pad down
                pad_down = max([pad_down,
                                corner_location[1] - dst_rows_num])
        panorama_cols_num = int(dst_cols_num + pad_right + pad_left)
        panorama_rows_num = int(dst_rows_num + pad_up + pad_down)
        pad_struct = PadStruct(pad_up=int(pad_up),
                               pad_down=int(pad_down),
                               pad_left=int(pad_left),
                               pad_right=int(pad_right))
        return panorama_rows_num, panorama_cols_num, pad_struct

    @staticmethod
    def add_translation_to_backward_homography(backward_homography: np.ndarray,
                                               pad_left: int,
                                               pad_up: int) -> np.ndarray:
        """Create a new homography which takes translation into account.

        Args:
            backward_homography: 3x3 Projective Homography matrix.
            pad_left: number of pixels that pad the destination image with
            zeros from left.
            pad_up: number of pixels that pad the destination image with
            zeros from the top.

        (1) Build the translation matrix from the pads.
        (2) Compose the backward homography and the translation matrix together.
        (3) Scale the homography as learnt in class.

        Returns:
            A new homography which includes the backward homography and the
            translation.
        """

        # translation matrix:
        translation_matrix = np.array([
            [1, 0, -pad_left],  # Move x-coordinates by -pad_left
            [0, 1, -pad_up],  # Move y-coordinates by -pad_up
            [0, 0, 1]  # Homogeneous coordinate remains unchanged
        ])

        # apply the translation to the homography mat
        final_homography = np.dot(backward_homography, translation_matrix)

        # normalize the homography
        final_homography /= final_homography[-1, -1]

        return final_homography
        pass

    def panorama(self,
                 src_image: np.ndarray,
                 dst_image: np.ndarray,
                 match_p_src: np.ndarray,
                 match_p_dst: np.ndarray,
                 inliers_percent: float,
                 max_err: float) -> np.ndarray:
        """Produces a panorama image from two images, and two lists of
        matching points, that deal with outliers using RANSAC.

        (1) Compute the forward homography and the panorama shape.
        (2) Compute the backward homography.
        (3) Add the appropriate translation to the homography so that the
        source image will plant in place.
        (4) Compute the backward warping with the appropriate translation.
        (5) Create the an empty panorama image and plant there the
        destination image.
        (6) place the backward warped image in the indices where the panorama
        image is zero.
        (7) Don't forget to clip the values of the image to [0, 255].


        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in pixels)
            between the mapped src point to its corresponding dst point,
            in order to be considered as valid inlier.

        Returns:
            A panorama image.

        """
        # step 1; compute forward homography
        forward_homography = self.compute_homography(match_p_src,
                                                     match_p_dst,
                                                     inliers_percent,
                                                     max_err)
        panorama_rows_num, panorama_cols_num, pad_struct = self.find_panorama_shape(src_image, dst_image,
                                                                                    forward_homography)

        # step 2; compute backward homography
        backward_homography = self.compute_homography(match_p_dst,
                                                      match_p_src,
                                                      inliers_percent,
                                                      max_err)

        # step 3; add the translation to the homography mat
        backward_homography = self.add_translation_to_backward_homography(backward_homography, pad_struct.pad_left,
                                                                          pad_struct.pad_up)

        # step 4; caculate the backward wrap
        img_panorama = self.compute_backward_mapping(backward_homography, src_image,
                                                     (panorama_rows_num, panorama_cols_num, 3))

        # step 5; place the destenation image in the img_panorama
        dst_rows, dst_cols, _ = dst_image.shape
        img_panorama[pad_struct.pad_up:pad_struct.pad_up + dst_rows, pad_struct.pad_left:pad_struct.pad_left + dst_cols] = dst_image

        return np.clip(img_panorama, 0, 255).astype(np.uint8)
        pass
