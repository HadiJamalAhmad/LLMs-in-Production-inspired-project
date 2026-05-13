


#!/usr/bin/env python  # Tells the operating system to run this file using the Python interpreter.
# coding: utf-8  # Declares that this file uses UTF-8 text encoding.

# In[4]:  # Marks this as cell number 4 from a Jupyter Notebook export.


import nltk  # Imports NLTK, a library for natural language processing tasks.
import numpy as np  # Imports NumPy for numerical arrays and matrix operations.
from utils import get_batches  # Imports a helper function that creates training batches.
from utils import compute_pca  # Imports a helper function that performs PCA dimensionality reduction.
from utils import get_dict  # Imports a helper function that creates word-index dictionaries.
import re  # Imports Python's regular expression library for text cleaning.
from matplotlib import pyplot  # Imports pyplot for plotting the final word embeddings.

# Create our corpus for training  # Explains that the next block loads the training text.
with open(r"C:\Users\HADI\Downloads\hamlet.txt", encoding="utf-8") as f:  # Opens the Hamlet text file using UTF-8 encoding.
    data = f.read()  # Reads the entire text file into the variable data.

# Slightly clean the data by removing punctuation, tokenizing by word, and converting to lowercase alpha characters  # Explains the preprocessing goal.
data = re.sub(r"[,!?;-]", ".", data)  # Replaces selected punctuation marks with periods.
data = nltk.word_tokenize(data)  # Splits the cleaned text into tokens using NLTK's word tokenizer.
data = [ch.lower() for ch in data if ch.isalpha() or ch == "."]  # Keeps alphabetic tokens and periods, then lowercases them.
print("Number of tokens:", len(data), "\n", data[500:515])  # Prints the token count and a small sample of tokens.

# Get our Bag of Words, along with a distribution  # Explains that the next lines compute token frequencies.
fdist = nltk.FreqDist(word for word in data)  # Creates a frequency distribution over all tokens in the corpus.
print("Size of vocabulary:", len(fdist))  # Prints the number of unique tokens.
print("Most Frequent Tokens:", fdist.most_common(20))  # Prints the 20 most frequent tokens.

# Create 2 dictionaries to speed up time-to-convert and keep track of vocabulary  # Explains that word-index mappings are created.
word2Ind, Ind2word = get_dict(data)  # Creates dictionaries mapping words to indices and indices to words.
V = len(word2Ind)  # Stores the vocabulary size in V.
print("Size of vocabulary:", V)  # Prints the vocabulary size.

print("Index of the word 'king':", word2Ind["king"])  # Prints the integer index assigned to the word king.
print("Word which has index 2743:", Ind2word[2743])  # Prints the word assigned to index 2743.


# Here we create our Neural network with 1 layer and 2 parameters  # Introduces model parameter initialization.
def initialize_model(N, V, random_seed=1):  # Defines a function to initialize weights and biases.
    """  # Starts the function documentation string.
    Inputs:  # Documents the input section.
        N: dimension of hidden vector  # Explains that N is the embedding or hidden-layer size.
        V: dimension of vocabulary  # Explains that V is the number of unique vocabulary tokens.
        random_seed: seed for consistent results in tests  # Explains that the seed makes random initialization reproducible.
    Outputs:  # Documents the output section.
        W1, W2, b1, b2: initialized weights and biases  # Explains that the function returns model parameters.
    """  # Ends the function documentation string.
    np.random.seed(random_seed)  # Sets the random seed so results are repeatable.

    W1 = np.random.rand(N, V)  # Initializes the first weight matrix with shape N by V.
    W2 = np.random.rand(V, N)  # Initializes the second weight matrix with shape V by N.
    b1 = np.random.rand(N, 1)  # Initializes the first bias vector with shape N by 1.
    b2 = np.random.rand(V, 1)  # Initializes the second bias vector with shape V by 1.

    return W1, W2, b1, b2  # Returns all initialized parameters.


# Create our final classification layer, which makes all possibilities add up to 1  # Introduces the softmax output function.
def softmax(z):  # Defines the softmax function.
    """  # Starts the function documentation string.
    Inputs:  # Documents the input section.
        z: output scores from the hidden layer  # Explains that z contains raw model scores.
    Outputs:  # Documents the output section.
        yhat: prediction (estimate of y)  # Explains that yhat contains normalized probabilities.
    """  # Ends the function documentation string.
    yhat = np.exp(z) / np.sum(np.exp(z), axis=0)  # Converts raw scores into probabilities that sum to 1 per column.
    return yhat  # Returns the probability predictions.


# Define the behavior for moving forward through our model, along with an activation function  # Introduces forward propagation.
def forward_prop(x, W1, W2, b1, b2):  # Defines the forward propagation function.
    """  # Starts the function documentation string.
    Inputs:  # Documents the input section.
        x: average one-hot vector for the context  # Explains that x represents the input context words.
        W1,W2,b1,b2: weights and biases to be learned  # Explains that these are the trainable model parameters.
    Outputs:  # Documents the output section.
        z: output score vector  # Explains that z contains raw output scores.
    """  # Ends the function documentation string.
    h = W1 @ x + b1  # Computes the hidden layer linear transformation.
    h = np.maximum(0, h)  # Applies the ReLU activation function to the hidden layer.
    z = W2 @ h + b2  # Computes the output layer raw scores.
    return z, h  # Returns the raw scores and hidden activations.


# Define how we determine the distance between ground truth and model predictions  # Introduces the cost function.
def compute_cost(y, yhat, batch_size):  # Defines a function to compute the training cost.
    logprobs = np.multiply(np.log(yhat), y) + np.multiply(  # Computes the first part of the binary cross-entropy-style loss.
        np.log(1 - yhat), 1 - y  # Computes the second part of the binary cross-entropy-style loss.
    )  # Closes the multiline loss calculation.
    cost = -1 / batch_size * np.sum(logprobs)  # Averages and negates the log probabilities to get the cost.
    cost = np.squeeze(cost)  # Removes unnecessary dimensions from the cost value.
    return cost  # Returns the scalar cost.


# Define how we move backward through the model and collect gradients  # Introduces backpropagation.
def back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size):  # Defines the backpropagation function.
    """  # Starts the function documentation string.
    Inputs:  # Documents the input section.
        x:  average one hot vector for the context  # Explains that x is the input context representation.
        yhat: prediction (estimate of y)  # Explains that yhat is the model prediction.
        y:  target vector  # Explains that y is the true target output.
        h:  hidden vector (see eq. 1)  # Explains that h is the hidden-layer activation.
        W1, W2, b1, b2:  weights and biases  # Explains that these are the current model parameters.
        batch_size: batch size  # Explains that batch_size is the number of examples in the batch.
     Outputs:  # Documents the output section.
        grad_W1, grad_W2, grad_b1, grad_b2:  gradients of weights and biases  # Explains that gradients are returned.
    """  # Ends the function documentation string.
    l1 = np.dot(W2.T, yhat - y)  # Backpropagates the output error into the hidden layer.
    l1 = np.maximum(0, l1)  # Applies a ReLU-like operation to the hidden-layer gradient.
    grad_W1 = np.dot(l1, x.T) / batch_size  # Computes the gradient for W1.
    grad_W2 = np.dot(yhat - y, h.T) / batch_size  # Computes the gradient for W2.
    grad_b1 = np.sum(l1, axis=1, keepdims=True) / batch_size  # Computes the gradient for b1.
    grad_b2 = np.sum(yhat - y, axis=1, keepdims=True) / batch_size  # Computes the gradient for b2.

    return grad_W1, grad_W2, grad_b1, grad_b2  # Returns all gradients.


# Put it all together and train  # Introduces the full training loop.
def gradient_descent(data, word2Ind, N, V, num_iters, alpha=0.03):  # Defines the gradient descent training function.
    """  # Starts the function documentation string.
    This is the gradient_descent function  # Briefly describes the function.

      Inputs:  # Documents the input section.
        data:      text  # Explains that data contains the processed text corpus.
        word2Ind:  words to Indices  # Explains that word2Ind maps words to integer indices.
        N:         dimension of hidden vector  # Explains that N is the hidden-layer size.
        V:         dimension of vocabulary  # Explains that V is the vocabulary size.
        num_iters: number of iterations  # Explains how many training iterations to run.
     Outputs:  # Documents the output section.
        W1, W2, b1, b2:  updated matrices and biases  # Explains that trained parameters are returned.

    """  # Ends the function documentation string.
    W1, W2, b1, b2 = initialize_model(N, V, random_seed=8855)  # Initializes the model parameters with a fixed seed.
    batch_size = 128  # Sets the number of training examples per batch.
    iters = 0  # Initializes the iteration counter.
    C = 2  # Sets the context window size.
    for x, y in get_batches(data, word2Ind, V, C, batch_size):  # Loops through batches of context inputs and target outputs.
        z, h = forward_prop(x, W1, W2, b1, b2)  # Runs forward propagation for the current batch.
        yhat = softmax(z)  # Converts output scores into predicted probabilities.
        cost = compute_cost(y, yhat, batch_size)  # Computes the cost for the current batch.
        if (iters + 1) % 10 == 0:  # Checks whether this is every tenth iteration.
            print(f"iters: {iters+1} cost: {cost:.6f}")  # Prints the iteration number and cost.
        grad_W1, grad_W2, grad_b1, grad_b2 = back_prop(  # Computes gradients using backpropagation.
            x, yhat, y, h, W1, W2, b1, b2, batch_size  # Passes all required inputs to the backpropagation function.
        )  # Closes the multiline backpropagation call.
        W1 = W1 - alpha * grad_W1  # Updates W1 using gradient descent.
        W2 = W2 - alpha * grad_W2  # Updates W2 using gradient descent.
        b1 = b1 - alpha * grad_b1  # Updates b1 using gradient descent.
        b2 = b2 - alpha * grad_b2  # Updates b2 using gradient descent.
        iters += 1  # Increments the iteration counter.
        if iters == num_iters:  # Checks whether the requested number of iterations has been reached.
            break  # Stops the training loop.
        if iters % 100 == 0:  # Checks whether 100 iterations have passed.
            alpha *= 0.66  # Reduces the learning rate to make later updates smaller.

    return W1, W2, b1, b2  # Returns the trained parameters.


# Train the model  # Introduces the model training section.
C = 2  # Sets the context window size.
N = 50  # Sets the hidden vector or embedding dimension.
word2Ind, Ind2word = get_dict(data)  # Recreates the word-to-index and index-to-word dictionaries.
V = len(word2Ind)  # Stores the vocabulary size.
num_iters = 150  # Sets the number of gradient descent iterations.
print("Call gradient_descent")  # Prints a message before training begins.
W1, W2, b1, b2 = gradient_descent(data, word2Ind, N, V, num_iters)  # Trains the model and stores the learned parameters.

# After listing 2.4 is done and gradient descent has been executed  # Introduces the embedding visualization step.
words = [  # Starts a list of selected words to visualize.
    "king",  # Adds king to the visualization list.
    "queen",  # Adds queen to the visualization list.
    "lord",  # Adds lord to the visualization list.
    "man",  # Adds man to the visualization list.
    "woman",  # Adds woman to the visualization list.
    "prince",  # Adds prince to the visualization list.
    "ophelia",  # Adds ophelia to the visualization list.
    "rich",  # Adds rich to the visualization list.
    "happy",  # Adds happy to the visualization list.
]  # Ends the list of selected words.
embs = (W1.T + W2) / 2.0  # Combines input and output embeddings by averaging them.
idx = [word2Ind[word] for word in words]  # Converts selected words into their corresponding indices.
X = embs[idx, :]  # Extracts the embedding vectors for the selected words.
print(X.shape, idx)  # Prints the shape of the selected embedding matrix and the selected indices.

result = compute_pca(X, 2)  # Reduces the selected word embeddings to 2 dimensions using PCA.
pyplot.scatter(result[:, 0], result[:, 1])  # Creates a 2D scatter plot of the reduced embeddings.
for i, word in enumerate(words):  # Loops through each selected word and its position.
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))  # Labels each point in the scatter plot with its word.
pyplot.show()  # Displays the final plot.




