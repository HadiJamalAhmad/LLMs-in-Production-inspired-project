import numpy as np  # Import NumPy for arrays, matrix multiplication, square roots, random numbers, and masking.
from scipy.special import softmax, logsumexp  # Import softmax for attention weights and logsumexp for numerically stable softmax.

print("Step 1: Input : 3 inputs, d_model=4")  # Print a label explaining that the input has 3 tokens/vectors and model dimension 4.
x = np.array(  # Create the input matrix x as a NumPy array.
    [[1.0, 0.0, 1.0, 0.0], [0.0, 2.0, 0.0, 2.0], [1.0, 1.0, 1.0, 1.0]]  # Define 3 input vectors, each with 4 features.
)  # Close the NumPy array construction.
print("x:", x)  # Print the input matrix x.

print("Step 2: weights 3 dimensions x d_model=4")  # Print a label explaining that the projection weights map d_model=4 to 3 dimensions.
w_query = np.array([[1, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 1]])  # Define the query projection matrix with shape 4 by 3.
print("w_query:", w_query)  # Print the query weight matrix.

w_key = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 0], [1, 1, 0]])  # Define the key projection matrix with shape 4 by 3.
print("w_key:", w_key)  # Print the key weight matrix.

w_value = np.array([[0, 2, 0], [0, 3, 0], [1, 0, 3], [1, 1, 0]])  # Define the value projection matrix with shape 4 by 3.
print("w_value:", w_value)  # Print the value weight matrix.

print("Step 3: Matrix multiplication to obtain Q,K,V")  # Print a label explaining that Q, K, and V will be computed.
print("Query: x * w_query")  # Print a label for the query calculation.
Q = np.matmul(x, w_query)  # Multiply x by w_query to produce the query matrix Q.
print("Q:", Q)  # Print the query matrix Q.

print("Key: x * w_key")  # Print a label for the key calculation.
K = np.matmul(x, w_key)  # Multiply x by w_key to produce the key matrix K.
print("K:", K)  # Print the key matrix K.

print("Value: x * w_value")  # Print a label for the value calculation.
V = np.matmul(x, w_value)  # Multiply x by w_value to produce the value matrix V.
print("V:", V)  # Print the value matrix V.

print("Step 4: Scaled Attention Scores")  # Print a label explaining that scaled dot-product attention scores will be computed.
k_d = np.sqrt(Q.shape[-1])  # Compute the square root of the query/key dimension for scaling.
attention_scores = (Q @ K.transpose()) / k_d  # Compute Q times K transposed, then divide by the scaling factor.
print(attention_scores)  # Print the raw scaled attention score matrix.

print("Step 5: Scaled softmax attention_scores for each vector")  # Print a label explaining that softmax will be applied to each row.
attention_scores[0] = softmax(attention_scores[0])  # Apply softmax to the first row of attention scores.
attention_scores[1] = softmax(attention_scores[1])  # Apply softmax to the second row of attention scores.
attention_scores[2] = softmax(attention_scores[2])  # Apply softmax to the third row of attention scores.
print(attention_scores[0])  # Print the first normalized attention row.
print(attention_scores[1])  # Print the second normalized attention row.
print(attention_scores[2])  # Print the third normalized attention row.

attention_scores = softmax(attention_scores, axis=1)  # Apply softmax again across each row of the attention score matrix.
print(attention_scores)  # Print the twice-normalized attention score matrix.

print("Step 6: attention value obtained by score1/k_d * V")  # Print a label explaining that weighted value vectors will be computed.
print(V[0])  # Print the first value vector.
print(V[1])  # Print the second value vector.
print(V[2])  # Print the third value vector.
attention1 = attention_scores[0].reshape(-1, 1)  # Reshape the first attention row into a column vector, though this is overwritten next.
attention1 = attention_scores[0][0] * V[0]  # Multiply the first attention weight by the first value vector.
print("Attention 1:", attention1)  # Print the first weighted value vector.

attention2 = attention_scores[0][1] * V[1]  # Multiply the second attention weight by the second value vector.
print("Attention 2:", attention2)  # Print the second weighted value vector.

attention3 = attention_scores[0][2] * V[2]  # Multiply the third attention weight by the third value vector.
print("Attention 3:", attention3)  # Print the third weighted value vector.

print(  # Start printing a multi-line label for the next step.
    "Step 7: sum the results to create the first line of the output matrix"  # Explain that weighted value vectors are summed.
)  # Close the print statement.
attention_input1 = attention1 + attention2 + attention3  # Sum the three weighted value vectors to get the first output row.
print(attention_input1)  # Print the first attention output vector.

print("Step 8: Step 1 to 7 for inputs 1 to 3")  # Print a label explaining that the same process would be repeated for all inputs.
# This assumes that we actually went through the whole process for all 3  # Explain the simplifying assumption.
# We'll just take a random matrix of the correct dimensions in lieu  # Explain that random values are used instead of computing all heads manually.
attention_head1 = np.random.random((3, 64))  # Create a random attention head output with shape 3 by 64.
print(attention_head1)  # Print the random attention head output.

print("Step 9: We assume we trained the 8 heads of the attention sub-layer")  # Print a label explaining that eight attention heads are assumed.
z0h1 = np.random.random((3, 64))  # Create a random output matrix for attention head 1.
z1h2 = np.random.random((3, 64))  # Create a random output matrix for attention head 2.
z2h3 = np.random.random((3, 64))  # Create a random output matrix for attention head 3.
z3h4 = np.random.random((3, 64))  # Create a random output matrix for attention head 4.
z4h5 = np.random.random((3, 64))  # Create a random output matrix for attention head 5.
z5h6 = np.random.random((3, 64))  # Create a random output matrix for attention head 6.
z6h7 = np.random.random((3, 64))  # Create a random output matrix for attention head 7.
z7h8 = np.random.random((3, 64))  # Create a random output matrix for attention head 8.
print("shape of one head", z0h1.shape, "dimension of 8 heads", 64 * 8)  # Print one head shape and total concatenated dimension.

print(  # Start printing a multi-line label for the concatenation step.
    "Step 10: Concatenate heads 1 to 8 to get the original 8x64=512 output dim"  # Explain the multi-head concatenation dimension.
)  # Close the print statement.
output_attention = np.hstack(  # Horizontally concatenate all eight attention head outputs.
    (z0h1, z1h2, z2h3, z3h4, z4h5, z5h6, z6h7, z7h8)  # Provide the eight matrices to concatenate.
)  # Close the horizontal stack operation.
print(output_attention)  # Print the final concatenated multi-head attention output.


def DotProductAttention(query, key, value, mask, scale=True):  # Define a function for dot-product attention.
    """Dot product self-attention."""  # Provide a short function docstring.
    # Args:  # Start documenting function arguments.
    #     query: array of query representations with shape (L_q by d)  # Explain the query input shape.
    #     key: array of key representations with shape (L_k by d)  # Explain the key input shape.
    #     value: array of value representations with shape (L_k by d) where L_v = L_k  # Explain the value input shape.
    #     mask: attention-mask, gates attention with shape (L_q by L_k)  # Explain the mask input shape and role.
    #     scale: whether to scale the dot product of the query and transposed key  # Explain the scaling flag.
    # Returns:  # Start documenting the return value.
    #     numpy.ndarray: Self-attention array for q, k, v arrays. (L_q by L_k)  # Explain the returned attention output.

    assert (  # Start an assertion to check that q, k, and v have matching embedding dimensions.
        query.shape[-1] == key.shape[-1] == value.shape[-1]  # Compare the last dimension of query, key, and value.
    ), "Embedding dimensions of q, k, v aren't all the same"  # Raise this message if the dimensions do not match.

    # Save dimension of the query embedding for scaling down the dot product  # Explain why depth is needed.
    if scale:  # Check whether scaling should be applied.
        depth = query.shape[-1]  # Use the embedding dimension as the scaling depth.
    else:  # Handle the case where scaling is disabled.
        depth = 1  # Use 1 so division by sqrt(depth) does not change the dot products.

    # Calculate scaled query key dot product according to formula above  # Explain the attention-score calculation.
    dots = np.matmul(query, np.swapaxes(key, -1, -2)) / np.sqrt(depth)  # Compute scaled dot products between query and key.

    # Apply the mask  # Explain that invalid attention positions may be blocked.
    if mask is not None:  # Check whether a mask was provided.
        dots = np.where(mask, dots, np.full_like(dots, -1e9))  # Keep allowed scores and replace masked scores with a very negative value.

    # Softmax formula implementation  # Explain that logsumexp is used to normalize scores stably.
    logsumexpo = logsumexp(dots, axis=-1, keepdims=True)  # Compute log-sum-exp over the last axis while preserving dimensions.

    # Take exponential of dots minus logsumexp to get softmax  # Explain the stable softmax conversion.
    dots = np.exp(dots - logsumexpo)  # Convert attention scores into normalized attention weights.

    # Multiply dots by value to get self-attention  # Explain the weighted sum of value vectors.
    attention = np.matmul(dots, value)  # Multiply attention weights by values to produce the attention output.

    return attention  # Return the attention output.


def masked_dot_product_self_attention(q, k, v, scale=True):  # Define a helper function for masked dot-product self-attention.
    """Masked dot product self attention."""  # Provide a short function docstring.
    # Args:  # Start documenting function arguments.
    #     q: queries.  # Explain the query tensor.
    #     k: keys.  # Explain the key tensor.
    #     v: values.  # Explain the value tensor.
    # Returns:  # Start documenting the return value.
    #     numpy.ndarray: masked dot product self attention tensor.  # Explain the returned masked attention tensor.

    # Size of the penultimate dimension of the query  # Explain that this dimension is used to build the attention mask.
    mask_size = q.shape[-2]  # Get the sequence length from the second-to-last dimension of q.

    # Creates ones below the diagonal and 0s above shape (1, mask_size, mask_size)  # Explain the causal lower-triangular mask.
    mask = np.tril(np.ones((1, mask_size, mask_size), dtype=np.bool_), k=0)  # Create a boolean lower-triangular mask.

    return DotProductAttention(q, k, v, mask, scale=scale)  # Return masked dot-product attention using the generated mask.