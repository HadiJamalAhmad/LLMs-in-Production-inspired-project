import os  # Import os so the script can create directories and check whether paths exist.
import torch  # Import PyTorch, the deep learning framework used to build and train the GPT model.
from accelerate import Accelerator  # Import Accelerator to simplify device placement and training setup.

import bitsandbytes as bnb  # Comment this out if running on Windows  # Import bitsandbytes for optional memory-efficient optimizers.


# Define the overall GPT Architecture  # Introduce the main GPT model class.
class GPT(torch.nn.Module):  # Define a GPT model by subclassing PyTorch's base neural-network Module class.
    def __init__(self):  # Define the model initialization method.
        super().__init__()  # Initialize the parent torch.nn.Module class.
        self.token_embedding = torch.nn.Embedding(vocab_size, n_embed)  # Create token embeddings that map token IDs to dense vectors.
        self.positional_embedding = torch.nn.Embedding(block_size, n_embed)  # Create position embeddings for each position in the context window.
        self.blocks = torch.nn.Sequential(  # Create a sequential stack of transformer blocks.
            *[Block(n_embed, n_head=n_head) for _ in range(n_layer)]  # Create n_layer transformer blocks and unpack them into the Sequential container.
        )  # Finish creating the transformer block stack.
        self.ln_f = torch.nn.LayerNorm(n_embed)  # Add the final layer normalization before the output head.
        self.lm_head = torch.nn.Linear(n_embed, vocab_size)  # Add the language-modeling head that projects embeddings to vocabulary logits.

        self.apply(self._init_weights)  # Apply custom weight initialization to all submodules.

    def forward(self, idx, targets=None):  # Define the forward pass, optionally receiving target tokens for loss computation.
        B, T = idx.shape  # Read the batch size B and sequence length T from the input tensor shape.

        tok_emb = self.token_embedding(idx)  # Convert input token IDs into token embedding vectors.
        pos_emb = self.positional_embedding(torch.arange(T, device=device))  # Create position embeddings for positions 0 through T-1 on the active device.
        x = tok_emb + pos_emb  # Add token and positional embeddings to create the transformer input.
        x = self.blocks(x)  # Pass the embedded sequence through the transformer blocks.
        x = self.ln_f(x)  # Apply the final layer normalization.
        logits = self.lm_head(x)  # Convert hidden states into vocabulary logits.

        if targets is None:  # Check whether target labels were not provided.
            loss = None  # Set loss to None when the model is used only for inference/generation.
        else:  # Handle the training/evaluation case where targets are provided.
            B, T, C = logits.shape  # Read batch size, sequence length, and vocabulary-size dimension from logits.
            logits = logits.view(B * T, C)  # Flatten logits so cross-entropy sees all positions as separate predictions.
            targets = targets.view(B * T)  # Flatten targets so they align with the flattened logits.
            loss = torch.nn.functional.cross_entropy(logits, targets)  # Compute cross-entropy loss between predicted logits and target token IDs.

        return logits, loss  # Return the raw logits and the optional loss.

    def _init_weights(self, module):  # Define the custom initialization function applied to model submodules.
        if isinstance(module, torch.nn.Linear):  # Check whether the submodule is a Linear layer.
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # Initialize Linear weights from a normal distribution.
            if module.bias is not None:  # Check whether the Linear layer has a bias vector.
                torch.nn.init.zeros_(module.bias)  # Initialize the Linear bias values to zero.
        elif isinstance(module, torch.nn.Embedding):  # Check whether the submodule is an Embedding layer.
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # Initialize Embedding weights from a normal distribution.

    def generate(self, idx, max_new_tokens):  # Define autoregressive token generation.
        for _ in range(max_new_tokens):  # Repeat generation for the requested number of new tokens.
            idx_cond = idx[:, -block_size:]  # Keep only the latest block_size tokens as context.
            logits, loss = self(idx_cond)  # Run the model on the current context.
            logits = logits[:, -1, :]  # Keep only the logits for the final time step.
            probs = torch.nn.functional.softmax(logits, dim=-1)  # Convert logits into probabilities over the vocabulary.
            idx_next = torch.multinomial(probs, num_samples=1)  # Sample the next token ID from the probability distribution.
            idx = torch.cat((idx, idx_next), dim=1)  # Append the sampled token to the running sequence.
        return idx  # Return the full generated token sequence.


# Define the building blocks of the model  # Introduce the transformer block components.
class Block(torch.nn.Module):  # Define one transformer block.
    def __init__(self, n_embed, n_head):  # Initialize the block with embedding size and number of attention heads.
        super().__init__()  # Initialize the parent torch.nn.Module class.
        head_size = n_embed // n_head  # Compute the dimensionality of each attention head.
        self.self_attention = MultiHeadAttention(n_head, head_size)  # Create the multi-head self-attention layer.
        self.feed_forward = FeedFoward(n_embed)  # Create the feed-forward network layer.
        self.ln1 = torch.nn.LayerNorm(n_embed)  # Create the first layer normalization before attention.
        self.ln2 = torch.nn.LayerNorm(n_embed)  # Create the second layer normalization before feed-forward processing.

    def forward(self, x):  # Define the forward pass through one transformer block.
        x = x + self.self_attention(self.ln1(x))  # Apply layer norm, self-attention, and a residual connection.
        x = x + self.feed_forward(self.ln2(x))  # Apply layer norm, feed-forward network, and a residual connection.
        return x  # Return the transformed hidden states.


class MultiHeadAttention(torch.nn.Module):  # Define the multi-head attention module.
    def __init__(self, num_heads, head_size):  # Initialize multi-head attention with the number of heads and each head size.
        super().__init__()  # Initialize the parent torch.nn.Module class.
        self.heads = torch.nn.ModuleList(  # Store attention heads in a ModuleList so PyTorch tracks their parameters.
            [Head(head_size) for _ in range(num_heads)]  # Create one Head module for each attention head.
        )  # Finish the ModuleList construction.
        self.projection = torch.nn.Linear(head_size * num_heads, n_embed)  # Project concatenated head outputs back to n_embed dimensions.
        self.dropout = torch.nn.Dropout(dropout)  # Create dropout for regularization after projection.

    def forward(self, x):  # Define the multi-head attention forward pass.
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # Run all heads and concatenate their outputs along the embedding dimension.
        out = self.dropout(self.projection(out))  # Project the concatenated output and apply dropout.
        return out  # Return the multi-head attention output.


class Head(torch.nn.Module):  # Define a single self-attention head.
    def __init__(self, head_size):  # Initialize one attention head with its output size.
        super().__init__()  # Initialize the parent torch.nn.Module class.
        self.key = torch.nn.Linear(n_embed, head_size, bias=False)  # Create the key projection without bias.
        self.query = torch.nn.Linear(n_embed, head_size, bias=False)  # Create the query projection without bias.
        self.value = torch.nn.Linear(n_embed, head_size, bias=False)  # Create the value projection without bias.
        self.register_buffer(  # Register a tensor buffer that moves with the model but is not a trainable parameter.
            "tril", torch.tril(torch.ones(block_size, block_size))  # Create a lower-triangular causal mask.
        )  # Finish registering the causal mask buffer.

        self.dropout = torch.nn.Dropout(dropout)  # Create dropout for attention probabilities.

    def forward(self, x):  # Define the forward pass for one attention head.
        _, T, _ = x.shape  # Read the sequence length T from the input tensor shape.
        k = self.key(x)  # Compute key vectors from the input.
        q = self.query(x)  # Compute query vectors from the input.
        attention = q @ k.transpose(-2, -1) * k.shape[-1] ** 0.5  # Compute scaled attention scores from queries and keys.
        attention = attention.masked_fill(  # Apply a mask to prevent attention to future tokens.
            self.tril[:T, :T] == 0, float("-inf")  # Set masked future-position scores to negative infinity.
        )  # Finish applying the causal mask.
        attention = torch.nn.functional.softmax(attention, dim=-1)  # Convert attention scores into attention probabilities.
        attention = self.dropout(attention)  # Apply dropout to attention probabilities.

        v = self.value(x)  # Compute value vectors from the input.
        out = attention @ v  # Weight the values by the attention probabilities.
        return out  # Return the attention-head output.


class FeedFoward(torch.nn.Module):  # Define the feed-forward network used inside each transformer block.
    def __init__(self, n_embed):  # Initialize the feed-forward network with the embedding size.
        super().__init__()  # Initialize the parent torch.nn.Module class.
        self.net = torch.nn.Sequential(  # Create a sequential feed-forward network.
            torch.nn.Linear(n_embed, 4 * n_embed),  # Expand the embedding dimension by a factor of 4.
            torch.nn.ReLU(),  # Apply a ReLU nonlinearity.
            torch.nn.Linear(4 * n_embed, n_embed),  # Project the expanded representation back to n_embed dimensions.
            torch.nn.Dropout(dropout),  # Apply dropout for regularization.
        )  # Finish the feed-forward network.

    def forward(self, x):  # Define the feed-forward forward pass.
        return self.net(x)  # Apply the feed-forward network to the input.


# Helper functions for training  # Introduce helper functions used by the training loop.
def encode(string):  # Define a function that converts text characters into integer IDs.
    return [utt2int[c] for c in string]  # Map each character in the string to its integer ID.


def decode(line):  # Define a function that converts integer IDs back into text.
    return "".join([int2utt[i] for i in line])  # Map each integer ID back to a character and join the characters.


def get_batch(split):  # Define a function that creates a training or validation batch.
    data = train_data if split == "train" else val_data  # Choose the training data or validation data based on the split argument.
    idx = torch.randint(len(data) - block_size, (batch_size,))  # Randomly sample starting indices for batch sequences.
    x = torch.stack([data[i : i + block_size] for i in idx])  # Build input sequences of length block_size.
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in idx])  # Build target sequences shifted one token ahead.
    x, y = x.to(device), y.to(device)  # Move input and target tensors to the active device.
    return x, y  # Return the input and target batch.


@torch.no_grad()  # Disable gradient tracking for the loss-estimation function.
def estimate_loss():  # Define a function that estimates average train and validation loss.
    out = {}  # Create a dictionary to store losses for each split.
    model.eval()  # Put the model in evaluation mode.
    for split in ["train", "val"]:  # Loop over the training and validation splits.
        losses = torch.zeros(eval_iters)  # Create a tensor to store loss values across evaluation iterations.
        for k in range(eval_iters):  # Run multiple evaluation batches.
            X, Y = get_batch(split)  # Get one batch from the current split.
            logits, loss = model(X, Y)  # Run the model and compute loss on the batch.
            losses[k] = loss.item()  # Store the scalar loss value.
        out[split] = losses.mean()  # Store the mean loss for this split.
    model.train()  # Return the model to training mode.
    return out  # Return the dictionary of estimated losses.


# Train the model  # Introduce the executable training section.
if __name__ == "__main__":  # Run the training code only when this file is executed directly.
    # Parmeters for our experiment  # Introduce experiment hyperparameters.
    batch_size = 64  # Number of utterances at once  # Set how many sequences are processed per batch.
    block_size = 256  # Maximum context window size  # Set the maximum sequence length/context window.
    max_iters = 5000  # Set the total number of training iterations.
    eval_interval = 500  # Set how often to evaluate train and validation loss.
    learning_rate = 3e-4  # Set the optimizer learning rate.
    eval_iters = 200  # Set how many batches are averaged during loss estimation.
    n_embed = 384  # Set the embedding dimension.
    n_head = 6  # Set the number of attention heads.
    n_layer = 6  # Set the number of transformer blocks.
    dropout = 0.2  # Set the dropout probability.
    accelerator = Accelerator()  # Create an Accelerator object for device/distributed training management.
    device = accelerator.device  # Get the active device selected by Accelerate.
    doing_quantization = False  # Change to True if imported bitsandbytes  # Decide whether to use the bitsandbytes optimizer.

    # Dataset  # Introduce dataset loading.
    with open("./crimeandpunishment.txt", "r", encoding="utf-8") as f:  # Open the training text file using UTF-8 encoding.
        text = f.read()  # Read the full text file into memory.

    # Character-based pseudo-tokenization  # Introduce character-level tokenization.
    chars = sorted(list(set(text)))  # Get all unique characters in the dataset and sort them.
    vocab_size = len(chars)  # Count the number of unique characters to define vocabulary size.
    utt2int = {ch: i for i, ch in enumerate(chars)}  # Build a character-to-integer mapping.
    int2utt = {i: ch for i, ch in enumerate(chars)}  # Build an integer-to-character mapping.

    data = torch.tensor(encode(text), dtype=torch.long)  # Encode the full text as integer IDs and convert it into a PyTorch tensor.
    n = int(0.9 * len(data))  # Compute the index for a 90/10 train-validation split.
    train_data = data[:n]  # Use the first 90 percent of the data for training.
    val_data = data[n:]  # Use the remaining 10 percent of the data for validation.

    # Instantiate the model and look at the parameters  # Introduce model creation and parameter counting.
    model = GPT().to(device)  # Create the GPT model and move it to the active device.
    print("Instantiated Model")  # Print a message confirming that the model was created.
    print(  # Start printing the model parameter count.
        sum(param.numel() for param in model.parameters()) / 1e6,  # Count trainable parameters in millions.
        "Model parameters",  # Print the label for the parameter count.
    )  # Finish printing the parameter count.

    optimizer = (  # Create the optimizer, optionally using bitsandbytes if quantization is enabled.
        torch.optim.AdamW(model.parameters(), lr=learning_rate)  # Use the standard AdamW optimizer when not quantizing.
        if not doing_quantization  # Check whether quantization is disabled.
        else bnb.optim.Adam(model.parameters(), lr=learning_rate)  # Use the bitsandbytes Adam optimizer when quantization is enabled.
    )  # Finish optimizer creation.
    print("Instantiated Optimizer")  # Print a message confirming that the optimizer was created.

    model, optimizer, train_data = accelerator.prepare(  # Let Accelerate prepare the model, optimizer, and training data.
        model, optimizer, train_data  # Pass the objects that Accelerate should prepare.
    )  # Finish Accelerate preparation.
    print("Prepared model, optimizer, and data")  # Print a message confirming Accelerate preparation.

    # Training block  # Introduce the main training loop.
    for iter in range(max_iters):  # Loop over all training iterations.
        print(f"Running Epoch {iter}")  # Print the current training iteration number.
        if iter % eval_interval == 0 or iter == max_iters - 1:  # Check whether it is time to evaluate losses.
            losses = estimate_loss()  # Estimate train and validation loss.
            print(  # Start printing the loss summary.
                f"| step {iter}: train loss {losses['train']:.4f} "  # Print the train loss for this step.
                "| validation loss {losses['val']:.4f} |"  # Print the validation loss template string.
            )  # Finish printing the loss summary.

        xb, yb = get_batch("train")  # Get one training batch.
        logits, loss = model(xb, yb)  # Run the model and compute training loss.
        optimizer.zero_grad(set_to_none=True)  # Clear existing gradients before backpropagation.
        accelerator.backward(loss)  # Backpropagate the loss through Accelerate.
        optimizer.step()  # Update model parameters using the optimizer.

    # Create model directory  # Introduce the model-output directory creation.
    model_dir = "./models/scratchGPT/"  # Define the directory where the trained model will be saved.
    if not os.path.exists(model_dir):  # Check whether the model directory does not exist yet.
        os.makedirs(model_dir)  # Create the model directory and any missing parent directories.

    # Save the model  # Introduce model saving.
    model_path = model_dir + "model.pt"  # Define the model checkpoint file path.
    torch.save(  # Start saving the model state dictionary.
        model.state_dict(),  # Save the model's learned parameters.
        model_path,  # Save the parameters to the checkpoint path.
    )  # Finish saving the model.

    # Load the saved model  # Introduce model loading.
    loaded_model = GPT().to(device)  # Create a fresh GPT model instance and move it to the active device.
    loaded_model.load_state_dict(torch.load(model_path))  # Load the saved parameters into the fresh model.

    # Test the loaded moel  # Introduce generation from the loaded model.
    context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Create a one-token starting context filled with token ID 0.
    print(  # Start printing generated text.
        decode(  # Decode the generated token IDs back into characters.
            loaded_model.generate(context, max_new_tokens=500)[0].tolist()  # Generate 500 new tokens and convert the first sequence to a Python list.
        )  # Finish decoding the generated tokens.
    )  # Finish printing the generated text.

# iedoloes own hawaehod it st iv ithaner, ye'ns soud bomg mo b hredan at  # Show an example generated text output.
# theng t'thed ond unyy ted wyy ; o bbyt." h eatourty at mere hevisall.on a  # Continue the example generated text output.
# odedect at heaAg Hme sgehed wer foutedr mas pvearouth ocqe  wato is f  # Continue the example generated text output.
# wave, 'lnto ran Tsun oo st ad s ce spit'tholint d pantulayoled I s  # Continue the example generated text output.
# asenois snt sked be heriseay aly mait ind t ft goveea ouriseants ces te"  # Finish the example generated text output.