import numpy as np


class SVD(object):
    def __init__(self):
        pass

    def svd(self, data):
        """
        Do SVD. You could use numpy SVD.
        Args:
                data: (N, D) TF-IDF features for the data.

        Return:
                U: (N,N) numpy array
                S: (min(N,D), ) numpy array
                V^T: (D,D) numpy array
        """

        U, S, Vh = np.linalg.svd(data, full_matrices=True, compute_uv=True)
        return U, S, Vh

    def rebuild_svd(self, U, S, V, k):
        """
        Rebuild SVD by k componments.

        Args:
                U: (N,N) numpy array
                S: (min(N,D), ) numpy array
                V: (D,D) numpy array
                k: int corresponding to number of components

        Return:
                data_rebuild: (N,D) numpy array

        Hint: numpy.matmul may be helpful for reconstruction.
        """
        
        # Create a diagonal matrix from the singular values
        S_diag = np.diag(S[:k])

        # Reconstruct the data
        data_rebuild = np.matmul(U[:, :k], np.matmul(S_diag, V[:k, :]))

        return data_rebuild

    def compression_ratio(self, data, k): 
        """
        Compute the compression ratio: (num stored values in compressed)/(num stored values in original)

        Args:
                data: (N, D) TF-IDF features for the data.
                k: int corresponding to number of components

        Return:
                compression_ratio: float of proportion of storage used
        """
        
        # Calculate the number of stored values in the original data
        num_stored_values_original = data.shape[0] * data.shape[1]

        # Calculate the number of stored values in the compressed data
        num_stored_values_compressed = (data.shape[1] * k) + k + (k * data.shape[0])
        
        # Calculate the compression ratio
        compression_ratio = num_stored_values_compressed / float(num_stored_values_original)

        return compression_ratio

    def recovered_variance_proportion(self, S, k):  
        """
        Compute the proportion of the variance in the original matrix recovered by a rank-k approximation

        Args:
                S: (min(N,D), ) numpy array
                k: int, rank of approximation

        Return:
                recovered_var: float corresponding to proportion of recovered variance
        """

        # Calculate the total variance in the original matrix
        total_variance = np.sum(S**2)

        # Calculate the variance recovered by the rank-k approximation
        recovered_variance = np.sum(S[:k]**2)

        # Calculate the proportion of the variance recovered
        recovered_variance_proportion = recovered_variance / total_variance

        return recovered_variance_proportion