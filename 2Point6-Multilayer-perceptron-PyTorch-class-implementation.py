#!/usr/bin/env python  # Use the system environment to locate the Python interpreter.
# coding: utf-8  # Declare UTF-8 encoding for this Python file.

# In[2]:  # Notebook cell marker showing this was originally cell 2.


import sys  # Import Python's sys module for interpreter/system information.
print(sys.executable)  # Print the path of the Python executable currently running this code.

import sys  # Import sys again; this is redundant but kept exactly from the original code.
print(sys.executable)  # Print the Python executable path again to confirm the same interpreter.

import numpy  # Import NumPy to confirm it is installed and available.
import torch  # Import PyTorch for tensors and neural network functionality.
print("NumPy OK")  # Print confirmation that NumPy imported successfully.
print("Torch OK")  # Print confirmation that PyTorch imported successfully.
import torch.nn as nn  # Import PyTorch neural network layers and modules under the alias nn.
import torch.nn.functional as F  # Import PyTorch functional operations, such as softmax.


class MultiLayerPerceptron(nn.Module):  # Define a multilayer perceptron class that inherits from PyTorch's nn.Module.
    def __init__(  # Define the constructor method used to initialize the model.
        self,  # Reference to the current model instance.
        input_size,  # Number of input features expected by the model.
        hidden_size=2,  # Number of neurons in each hidden layer, defaulting to 2.
        output_size=3,  # Number of output features/classes, defaulting to 3.
        num_hidden_layers=1,  # Number of hidden layers to create, defaulting to 1.
        hidden_activation=nn.Sigmoid,  # Activation layer class to use after each hidden linear layer.
    ):  # End of the constructor parameter list.
        """Initialize weights.  # Start the constructor docstring describing initialization.
        Args:  # Begin the arguments section of the docstring.
            input_size (int): size of the input  # Document the input_size argument.
            hidden_size (int): size of the hidden layers  # Document the hidden_size argument.
            output_size (int): size of the output  # Document the output_size argument.
            num_hidden_layers (int): number of hidden layers  # Document the num_hidden_layers argument.
            hidden_activation (torch.nn.*): the activation class  # Document the hidden_activation argument.
        """  # End the constructor docstring.
        super(MultiLayerPerceptron, self).__init__()  # Initialize the parent nn.Module class.
        self.module_list = nn.ModuleList()  # Create a ModuleList to store hidden layers and activations.
        interim_input_size = input_size  # Set the first layer's input size to the model input size.
        interim_output_size = hidden_size  # Set each hidden layer's output size to hidden_size.
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Create a device object depending on CUDA availability, but do not store or use it.

        for _ in range(num_hidden_layers):  # Loop once for each requested hidden layer.
            self.module_list.append(  # Append a new module to the hidden module list.
                nn.Linear(interim_input_size, interim_output_size)  # Add a linear layer from current input size to hidden size.
            )  # End the append call for the linear layer.
            self.module_list.append(hidden_activation())  # Add an instance of the chosen activation function.
            interim_input_size = interim_output_size  # Update the next layer's input size to match the hidden output size.

        self.fc_final = nn.Linear(interim_input_size, output_size)  # Define the final linear layer from hidden size to output size.

        self.last_forward_cache = []  # Create an empty cache list, although it is not used in the current forward pass.

    def forward(self, x, apply_softmax=False):  # Define the forward pass called when the model processes input data.
        """The forward pass of the MLP  # Start the forward-method docstring.

        Args:  # Begin the arguments section of the docstring.
            x_in (torch.Tensor): an input data tensor.  # Document the intended input tensor argument.
                x_in.shape should be (batch, input_dim)  # Document the expected input tensor shape.
            apply_softmax (bool): a flag for the softmax activation  # Document whether softmax should be applied.
                should be false if used with the Cross Entropy losses  # Explain that CrossEntropyLoss expects raw logits.
        Returns:  # Begin the return-value section of the docstring.
            the resulting tensor. tensor.shape should be (batch, output_dim)  # Document the expected output tensor shape.
        """  # End the forward-method docstring.
        for module in self.module_list:  # Iterate through every hidden layer and activation module.
            x = module(x)  # Apply the current module to the tensor and update x.

        output = self.fc_final(x)  # Apply the final linear layer to produce output logits.

        if apply_softmax:  # Check whether the caller requested softmax probabilities.
            output = F.softmax(output, dim=1)  # Apply softmax across the output dimension for each sample.

        return output  # Return the final output tensor.


# In[4]:  # Notebook cell marker showing this was originally cell 4.


import torch  # Import PyTorch again; this is redundant but kept exactly from the original code.

model = MultiLayerPerceptron(  # Create an instance of the MultiLayerPerceptron model.
    input_size=4,  # Configure the model to accept 4 input features.
    hidden_size=5,  # Configure each hidden layer to contain 5 neurons.
    output_size=3,  # Configure the model to output 3 values per sample.
    num_hidden_layers=2  # Configure the model to use 2 hidden layers.
)  # Finish creating the model instance.

x = torch.randn(2, 4)  # Create a random input tensor representing a batch of 2 samples with 4 features each.

y = model(x)  # Run the input tensor through the model to compute the output.

print("Input:", x)  # Print the randomly generated input tensor.
print("Output:", y)  # Print the model output tensor.


# In[ ]:  # Empty notebook cell marker.


# In[ ]:  # Empty notebook cell marker.