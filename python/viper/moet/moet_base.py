"""
Base class for MOE models.
"""

import numpy as np


class MOETBase(object):
    """
    Base class for Mixture of Experts (MOE) models.
    """

    def __init__(self, experts_no, default_type=np.float32):
        """

        Args:
            default_type:
        """
        self.experts_no = experts_no
        self.default_type = default_type

    def softmax(self, x_normalized, tetag, experts_no):
        """
        Args:
            x_normalized: Array of normalized feature vectors.
        """
        x = np.tile(x_normalized, experts_no) * tetag
        x = x.reshape(-1, x_normalized.shape[0], x_normalized.shape[1]).sum(
            axis=2).reshape(
            x_normalized.shape[0], -1)
        e_x = np.exp(
            x - np.array([np.max(x, axis=1), ] * x.shape[1]).transpose())
        out = (e_x.transpose() / e_x.sum(axis=1)).transpose()
        return out

    def h_fun(self, gating, pdf):
        h = ((gating * pdf).T / np.sum(gating * pdf, axis=1)).T
        return h

    def ds_dtetag_old(self, x_normalized, tetag, experts_no):
        """
        Args:
            x_normalized: Array of normalized feature vectors.
        """
        N = x_normalized.shape[0]
        U = tetag.shape[0]
        E = experts_no

        dsdtetag = np.zeros([N, U, E], dtype=self.default_type)
        no_x = int(U / E)
        for i in range(N):
            for j in range(E):
                dsdtetag[i, j * no_x : (j + 1) * no_x, j] = x_normalized[i, :]

        return dsdtetag

    def ds_dtetag(self, x_normalized, tetag, experts_no):
        """
        Args:
            x_normalized: Array of normalized feature vectors.
        """
        N = x_normalized.shape[0]
        U = tetag.shape[0]
        E = experts_no

        dsdtetag = np.zeros([N, U, E], dtype=self.default_type)
        no_x = int(U / E)

        for j in range(E):
            dsdtetag[:, j * no_x: (j + 1) * no_x, j] = x_normalized[:, :]

        return dsdtetag

    def e_fun_old(self, hf, gating, dsdtetag):
        """Unparallelized version."""
        # N - #examples, E - #experts
        # hf dims: (N, E)
        # Gating dims: (N, E);
        # dsdtetag dims: (N, U, E)
        N = dsdtetag.shape[0]
        E = dsdtetag.shape[2]
        e = 0
        for i in range(N):
            for j in range(E):
                e += (hf - gating)[i, j] * dsdtetag[i, :, j]
        return e

    def e_fun(self, hf, gating, dsdtetag):
        """Parallelized version."""

        # N - #examples, E - #experts
        # hf dims: (N, E)
        # Gating dims: (N, E);
        # dsdtetag dims: (N, U, E)
        N = dsdtetag.shape[0]
        U = dsdtetag.shape[1]
        E = dsdtetag.shape[2]

        # dims: (N, E)
        tmp1 = (hf - gating)
        # dims: (N * E)
        tmp1 = np.reshape(tmp1, (N * E))
        # dims: (N * E, 1)
        tmp1 = tmp1[:, np.newaxis]
        # No need for repeating, as broadcasting will be used.
        # dims: (N * E, U)
        # tmp1 = np.repeat(tmp1[:, np.newaxis], U, axis=1)

        # dims: (N, E, U)
        tmp2 = np.swapaxes(dsdtetag, 1, 2)
        # dims: (N * E, U)
        tmp2 = np.reshape(tmp2, (N * E, U))

        # tmp1 *= tmp2 does computation in place, and consumes less memory then tmp1 * tmp2.
        tmp2 *= tmp1
        e = np.sum(tmp2, axis=0)

        return e

    def R_fun_old(self, gating, dsdtetag, experts_no):
        """Unparallelized version."""

        # Gating dims: (N, E); N - #examples, E - #experts
        # dsdtetag dims: (N, U, E)
        # E = experts_no
        N = dsdtetag.shape[0]
        U = dsdtetag.shape[1]
        E = experts_no

        R = np.zeros([U, U])
        for i in range(N):
            for j in range(E):
                pom1 = gating[i, j] * (1 - gating[i, j])
                pom2 = np.expand_dims(dsdtetag[i, :, j], axis=1)
                # Multiply pom1 with outer product of pom2 and add to R.
                R += pom1 * (pom2.dot(pom2.T))
        return R
    
    def R_fun(self, gating, dsdtetag, experts_no):
        """
        Computes value of R function.

        Args:
          gating: Gating values, dimensions are (N, E); N - #examples, E - #experts.
          dsdtetag: ds/dteta_g values, dimensions are (N, U, E).
          experts_no: Number of experts.
        Returns:
          R value, dimensions are (U, U).
        """

        N = dsdtetag.shape[0]
        U = dsdtetag.shape[1]
        E = experts_no

        # TODO: Rewrite this (maybe modify fit function to do batching instead).
        # This is done in order to prevent Memory Errors.
        max_batch_size = 10000
        num_batches = np.ceil(float(N) / max_batch_size).astype(int)

        R = np.zeros([U, U], dtype=self.default_type)
        for batch_idx in range(num_batches):
            start_index = batch_idx * max_batch_size
            end_index = start_index + max_batch_size
            gating_sliced = gating[start_index:end_index, :]
            dsdtetag_sliced = dsdtetag[start_index:end_index, :, :]

            n = gating_sliced.shape[0]

            # Element-wise multiply
            # dims: (N, E)
            tmp1 = gating_sliced * (1 - gating_sliced)
            # dims: (N * E)
            tmp1 = np.reshape(tmp1, (n * E))
            # dims: (N * E, 1, 1)
            tmp1 = tmp1[:, np.newaxis, np.newaxis]
            # There is no need to repeat the array,
            # as broadcasting will be used (this is more memory efficient way).
            # dims: (N * E, U, U)
            # tmp1 = np.repeat(
            #     np.repeat(tmp1, U, axis=1),
            #     U, axis=2)

            # dims: (N, E, U)
            tmp2 = np.swapaxes(dsdtetag_sliced, 1, 2)
            # dims: (N * E, U)
            tmp2 = np.reshape(tmp2, (n * E, U))

            # https://stackoverflow.com/questions/48498662/numpy-row-wise-outer-product-of-two-matrices
            # dims: (N * E, U, U)
            tmp2 = np.matmul(tmp2[:, :, np.newaxis], tmp2[:, np.newaxis, :])
            # Alternative way to compute.
            # tmp2 = np.einsum('bi,bo->bio', tmp2, tmp2, optimize='greedy')

            # tmp2 *= tmp1 does computation in place, and consumes less memory then tmp1 * tmp2.
            tmp2 *= tmp1
            R += np.sum(tmp2, axis=0)

        return R
