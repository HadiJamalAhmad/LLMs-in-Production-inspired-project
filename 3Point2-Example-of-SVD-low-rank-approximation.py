import scipy  # Import the SciPy library, which provides scientific computing tools including sparse linear algebra.
import numpy as np  # Import NumPy and give it the short alias "np" for array and numerical operations.

matrix = np.array(  # Create a NumPy array from the nested list below.
    [  # Start the outer list representing the rows of the matrix.
        [1.0, 2.0, 3.0, 4.0],  # First row of the 4x4 matrix.
        [5.0, 6.0, 7.0, 8.0],  # Second row of the 4x4 matrix.
        [9.0, 10.0, 11.0, 12.0],  # Third row of the 4x4 matrix.
        [13.0, 14.0, 15.0, 16.0],  # Fourth row of the 4x4 matrix.
    ]  # End the outer list of matrix rows.
)  # Finish creating the NumPy array.
u, s, vt = scipy.sparse.linalg.svds(matrix, k=1)  # Compute the rank-1 sparse SVD: left singular vector u, singular value s, and right singular vector transpose vt.
print(u, s, vt)  # Print the SVD components.
# [[-0.1347221]  # Printed left singular vector u, row 1.
#  [-0.3407576]  # Printed left singular vector u, row 2.
#  [-0.5467932]  # Printed left singular vector u, row 3.
#  [-0.7528288]], [38.62266], [[-0.428412  -0.4743725 -0.5203326 -0.566292]]  # Printed left singular vector u row 4, singular value s, and right singular vector transpose vt.

svd_matrix = u * s * vt  # Reconstruct a rank-1 approximation of the original matrix using broadcasting: u multiplied by s and vt.
print(svd_matrix)  # Print the reconstructed rank-1 approximation matrix.
# array([[ 2.2291691,  2.4683154,  2.7074606,  2.9466066],  # First row of the reconstructed rank-1 matrix.
#        [ 5.6383204,  6.243202 ,  6.848081 ,  7.4529614],  # Second row of the reconstructed rank-1 matrix.
#        [ 9.047472 , 10.018089 , 10.988702 , 11.959317 ],  # Third row of the reconstructed rank-1 matrix.
#        [12.456624 , 13.792976 , 15.129323 , 16.465673 ]], dtype=float32)  # Fourth row of the reconstructed rank-1 matrix and its dtype.