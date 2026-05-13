# %%  # Marks a Jupyter/VS Code cell boundary so this script can be run cell-by-cell.
#!/usr/bin/env python  # Declares the Python interpreter when the file is executed as a script on Unix-like systems.
# coding: utf-8  # Declares UTF-8 source-code encoding for non-ASCII characters such as emojis.

# In[5]:  # Keeps the original notebook execution-cell marker from the exported notebook.


import sys  # Imports Python system utilities, including the current interpreter path.
print(sys.executable)  # Prints the exact Python executable being used for this run.

import copy  # Imports copy utilities so model weights can be deep-copied safely.
import torch  # Imports PyTorch for tensors, neural networks, losses, and optimization.
import pandas as pd  # Imports pandas for loading and manipulating tabular CSV data.
import numpy as np  # Imports NumPy for numeric arrays, permutations, and matrix operations.
from gensim.models import Word2Vec  # Imports Gensim Word2Vec to train word embeddings on the tweet corpus.
from sklearn.model_selection import train_test_split  # Imports the train/test splitting helper from scikit-learn.
import nltk  # Imports NLTK for tokenization tools and English stop words.
import spacy  # Imports spaCy for linguistic processing and lemmatization.
from sklearn.metrics import confusion_matrix, classification_report, f1_score  # Imports classification evaluation metrics.
from sklearn.preprocessing import label_binarize  # Imports label binarization for one-vs-rest ROC curves.
from sklearn.metrics import roc_curve, auc  # Imports ROC-curve generation and area-under-curve scoring.
import matplotlib.pyplot as plt  # Imports Matplotlib plotting functions for charts.
import seaborn as sns  # Imports Seaborn for heatmaps and nicer statistical visualizations.
try:  # Tries to use already-installed tokenizer resources, stop words, and spaCy model first.
    tokenizer = nltk.tokenize.RegexpTokenizer("\w+'?\w+|\w+'")  # Creates a regex tokenizer that keeps simple apostrophe-containing words.
    tokenizer.tokenize("This is a test")  # Tests that the tokenizer works before continuing.
    stop_words = nltk.corpus.stopwords.words("english")  # Loads the English stop-word list from NLTK.
    nlp = spacy.load("en_core_web_lg", disable=["parser", "ner"])  # Loads the large English spaCy model while disabling unused components.
except Exception:  # Runs this fallback when required NLTK data or spaCy model files are missing.
    nltk.download("stopwords")  # Downloads the NLTK stop-word corpus.
    nltk.download("punkt")  # Downloads NLTK punkt tokenizer resources for compatibility.
    spacy.cli.download("en_core_web_lg")  # Downloads the large English spaCy model.
    tokenizer = nltk.tokenize.RegexpTokenizer("\w+'?\w+|\w+'")  # Recreates the regex tokenizer after downloads complete.
    tokenizer.tokenize("This is a test")  # Tests tokenization again after installing resources.
    stop_words = nltk.corpus.stopwords.words("english")  # Reloads English stop words after downloading them.
    nlp = spacy.load("en_core_web_lg", disable=["parser", "ner"])  # Reloads spaCy with parser and NER disabled for faster lemmatization.

# Create our corpus for training and perform some classic NLP preprocessing  # Describes the following dataset-loading and preprocessing section.
print("✅ Loading dataset...")  # Prints a status message before loading the CSV file.
dataset = pd.read_csv(r"C:\Users\HADI\Downloads\twitter.csv")  # Loads the Twitter sentiment dataset from the local Windows path.
dataset = dataset.dropna(subset=["text"])  # Removes rows where the tweet text is missing.
print("Dataset shape:", dataset.shape)  # Prints the number of rows and columns in the dataset.
print("Columns:", dataset.columns)  # Prints the column names available in the CSV file.
print(dataset.head())  # Prints the first five rows for a quick inspection.

print("🔄 Tokenizing text...")  # Prints a status message before tokenization.
text_data = list(  # Converts the mapped tokenized output into a concrete list.
    map(  # Applies the same preprocessing function to every tweet.
        lambda x: [  # Defines an inline function that returns filtered tokens for one tweet.
            word for word in tokenizer.tokenize(x.lower())  # Lowercases the tweet and tokenizes it into words.
            if word not in stop_words  # Keeps only tokens that are not English stop words.
        ],  # Ends the token-filtering list comprehension.
        dataset["text"],  # Supplies the dataset text column as the input sequence.
    )  # Ends the map call.
)  # Ends the list conversion.
print("Sample tokenized text:", text_data[:2])  # Prints the first two tokenized examples.

print("🧠 Lemmatizing text (this may take time)...")  # Prints a status message before spaCy lemmatization.
docs = list(nlp.pipe([" ".join(text) for text in text_data], batch_size=256))  # Processes token lists as space-joined documents in efficient batches.
text_data = [[token.lemma_ for token in doc] for doc in docs]  # Replaces each token with its lemma for normalization.

print("sample Lemmatized text:", text_data[:2])  # Prints the first two lemmatized examples.


print("🏷️ Processing labels...")  # Prints a status message before label preparation.

# Remove empty token lists and keep labels aligned  # Explains why the next list comprehension pairs tokens with labels.
filtered_pairs = [  # Builds aligned token-label pairs while removing empty token sequences.
    (tokens, label)  # Stores one token list and its sentiment label together.
    for tokens, label in zip(text_data, dataset["sentiment"])  # Iterates through processed texts and original sentiment labels in parallel.
    if len(tokens) > 0  # Keeps only examples that still contain at least one token.
]  # Ends the filtered token-label pair list.

text_data = [tokens for tokens, _ in filtered_pairs]  # Extracts the cleaned token lists from the filtered pairs.

label_map = {  # Defines the mapping from sentiment strings to numeric class IDs.
    "negative": 0,  # Maps negative sentiment to class index 0.
    "neutral": 1,  # Maps neutral sentiment to class index 1.
    "positive": 2  # Maps positive sentiment to class index 2.
}  # Ends the sentiment-to-index mapping.
label_data = [label_map[label] for _, label in filtered_pairs]  # Converts string labels to numeric class IDs.

print("Sample labels:", label_data[:10])  # Prints the first ten numeric labels.
print("Number of samples:", len(label_data))  # Prints the number of usable labeled examples.

from collections import Counter  # Imports Counter to count labels per class.

print("\n📊 Class distribution:")  # Prints a heading for the class-distribution output.
class_counts = Counter(label_data)  # Counts how many examples belong to each class.
print(class_counts)  # Prints the class counts.

num_classes = 3  # Stores the number of sentiment classes.
total_samples = len(label_data)  # Stores the total number of filtered training examples.

class_weights = []  # Initializes a list for inverse-frequency class weights.
for class_id in range(num_classes):  # Iterates over each class index.
    class_count = class_counts[class_id]  # Gets the sample count for the current class.
    weight = total_samples / (num_classes * class_count)  # Computes a balanced class weight for the current class.
    class_weights.append(weight)  # Appends the computed weight to the list.

class_weights = torch.tensor(class_weights, dtype=torch.float)  # Converts class weights into a floating-point PyTorch tensor.

print("⚖️ Class weights:", class_weights)  # Prints the class weights used by the weighted loss function.

assert len(text_data) == len(  # Checks that every tokenized example still has exactly one label.
    label_data  # Supplies the label list length for the alignment assertion.
), f"{len(text_data)} does not equal {len(label_data)}"  # Raises a clear error if text and label counts differ.

EMBEDDING_DIM = 100  # Sets the Word2Vec embedding dimensionality to 100.

print("⚙️ Training Word2Vec...")  # Prints a status message before training embeddings.
model = Word2Vec(  # Creates and trains a Word2Vec model on the processed token sequences.
    text_data, vector_size=EMBEDDING_DIM, window=5, min_count=1, workers=4  # Sets corpus, vector size, context window, minimum count, and worker threads.
)  # Ends Word2Vec model construction and training.
print("✅ Word2Vec trained")  # Prints a success message after Word2Vec training.
word_vectors = model.wv  # Extracts the trained keyed vectors from the Word2Vec model.
print(f"Vocabulary Length: {len(model.wv)}")  # Prints the number of words in the learned vocabulary.
del model  # Deletes the full Word2Vec model object to save memory while keeping word vectors.

padding_value = len(word_vectors.index_to_key)  # Sets the initial padding index to one past the last vocabulary index.

# Embeddings are needed to give semantic value to the inputs of an LSTM  # Explains why embedding weights are built for the neural models.
print("🔢 Creating embedding weights...")  # Prints a status message before creating the embedding matrix.
embedding_weights = torch.Tensor(word_vectors.vectors)  # Converts Word2Vec vectors into a PyTorch tensor.

# Add padding vector (zeros)  # Explains that a zero vector is appended for padded positions.
pad_vector = torch.zeros(1, embedding_weights.shape[1])  # Creates one all-zero embedding row for padding tokens.
embedding_weights = torch.cat((embedding_weights, pad_vector), dim=0)  # Appends the padding vector to the embedding matrix.

padding_value = embedding_weights.shape[0] - 1  # Updates the padding index to the final row of the embedding matrix.
print("Embedding shape:", embedding_weights.shape)  # Prints the final embedding matrix shape.


class KhushuAttention(torch.nn.Module):  # Defines an attention module for summarizing LSTM outputs.
    def __init__(self, hidden_dim):  # Initializes the attention module with the hidden-state size.
        super().__init__()  # Initializes the parent torch.nn.Module class.
        self.attn = torch.nn.Linear(hidden_dim * 2, 1)  # Learns one scalar attention score per bidirectional hidden state.
        

    def forward(self, outputs):  # Defines the forward pass that converts sequence outputs into a context vector.
        # outputs: [seq_len, batch, hidden_dim*2]  # Documents the expected incoming tensor shape.
        outputs = outputs.permute(1, 0, 2)  # Reorders outputs to [batch, seq, hidden] for batch matrix multiplication.
        
        scores = self.attn(outputs).squeeze(-1)  # Produces one unnormalized attention score per token position.
        weights = torch.softmax(scores, dim=1)  # Normalizes scores across the sequence dimension into attention weights.

        context = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)  # Computes the weighted average context vector for each sequence.

        return context, weights  # Returns both the context vector and the attention weights for inspection.


class RNN(torch.nn.Module):  # Defines a simple RNN sentiment classifier.
    def __init__(  # Starts the constructor for the RNN model.
        self,  # Refers to the model instance being initialized.
        input_dim,  # Receives the vocabulary size or input dimension.
        embedding_dim,  # Receives the dimension of each embedding vector.
        hidden_dim,  # Receives the number of hidden units in the RNN.
        output_dim,  # Receives the number of output sentiment classes.
        embedding_weights,  # Receives the pretrained embedding matrix.
    ):  # Ends the constructor parameter list.
        super().__init__()  # Initializes the parent torch.nn.Module class.
        self.embedding = torch.nn.Embedding.from_pretrained(  # Creates an embedding layer initialized with Word2Vec weights.
        embedding_weights,  # Supplies the pretrained embedding matrix.
        padding_idx=padding_value,  # Tells PyTorch which embedding row represents padding.
        freeze=True  # Freezes embeddings so Word2Vec vectors are not updated during training.
        )  # Ends embedding-layer creation.
        self.rnn = torch.nn.RNN(embedding_dim, hidden_dim)  # Creates a vanilla RNN over the embedded token sequence.
        self.fc = torch.nn.Linear(hidden_dim, output_dim)  # Maps the final RNN hidden state to class logits.

    def forward(self, x, text_lengths):  # Defines how one padded batch flows through the RNN model.
        embedded = self.embedding(x)  # Converts token indices into embedding vectors.
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(  # Packs padded sequences so the RNN ignores padding positions.
             embedded,  # Supplies embedded sequence data shaped as sequence-first tensors.
             text_lengths.cpu(),  # Supplies true sequence lengths on CPU as required by packing.
             enforce_sorted=False  # Allows batches that are not sorted by sequence length.
    )  # Ends padded-sequence packing.
        packed_output, hidden = self.rnn(packed_embedded)  # Runs the packed embeddings through the RNN and returns packed outputs plus hidden state.
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(  # Unpacks RNN outputs back into padded tensor form.
            packed_output  # Supplies the packed output returned by the RNN.
        )  # Ends unpacking of the packed RNN output.
        return self.fc(hidden.squeeze(0)), None  # Returns class logits and None because this model has no attention weights.


INPUT_DIM = padding_value  # Sets the input vocabulary dimension based on the padding index value.
EMBEDDING_DIM = 100  # Sets the embedding dimension used by the RNN model.
HIDDEN_DIM = 256  # Sets the RNN hidden-state dimension.
OUTPUT_DIM = 3  # Sets the number of sentiment classes.

rnn_model = RNN(  # Instantiates the vanilla RNN classifier.
    INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, embedding_weights  # Passes model dimensions and pretrained embedding weights.
)  # Ends RNN model instantiation.

rnn_optimizer = torch.optim.SGD(rnn_model.parameters(), lr=1e-3)  # Creates an SGD optimizer for the RNN model.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Selects GPU if available, otherwise CPU.
class_weights = class_weights.to(device)  # Moves class weights to the selected compute device.
rnn_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)  # Creates weighted cross-entropy loss for RNN training.

class LSTM(torch.nn.Module):  # Defines a bidirectional LSTM sentiment classifier with attention.
    def __init__(  # Starts the constructor for the LSTM model.
        self,  # Refers to the model instance being initialized.
        input_dim,  # Receives the vocabulary size or input dimension.
        embedding_dim,  # Receives the dimension of each word embedding.
        hidden_dim,  # Receives the number of LSTM hidden units per direction.
        output_dim,  # Receives the number of sentiment classes.
        n_layers,  # Receives the number of stacked LSTM layers.
        bidirectional,  # Receives whether the LSTM should run forward and backward.
        dropout,  # Receives the dropout probability.
        embedding_weights,  # Receives the pretrained embedding matrix.
    ):  # Ends the constructor parameter list.
        super().__init__()  # Initializes the parent torch.nn.Module class.
        self.embedding = torch.nn.Embedding.from_pretrained(  # Creates an embedding layer initialized from Word2Vec.
        embedding_weights,  # Supplies pretrained embedding weights.
        padding_idx=padding_value,  # Specifies which index is used for padding.
        freeze=True  # Freezes the embeddings during supervised model training.
        )  # Ends embedding-layer creation.
        self.rnn = torch.nn.LSTM(  # Creates the LSTM sequence encoder.
            embedding_dim,  # Sets the input feature size to the embedding dimension.
            hidden_dim,  # Sets the hidden size per LSTM direction.
            num_layers=n_layers,  # Sets the number of stacked LSTM layers.
            bidirectional=bidirectional,  # Enables or disables bidirectional processing.
            dropout=dropout,  # Applies dropout between LSTM layers when n_layers > 1.
        )  # Ends LSTM layer creation.
        self.fc = torch.nn.Linear(hidden_dim * 2, output_dim)  # Maps bidirectional attention context to output logits.
        self.dropout = torch.nn.Dropout(dropout)  # Creates a dropout layer, although it is not explicitly used in forward here.
        self.attention = KhushuAttention(hidden_dim)  # Creates the custom attention module for LSTM outputs.
        
    def forward(self, x, text_lengths):  # Defines how one padded batch flows through the LSTM model.
        embedded = self.embedding(x)  # Converts token indices into embedding vectors.
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(  # Packs variable-length embedded sequences for efficient LSTM processing.
    embedded, text_lengths.cpu(), enforce_sorted=False  # Supplies embeddings, CPU lengths, and permits unsorted batches.
        )  # Ends padded-sequence packing.
        packed_output, (hidden, cell) = self.rnn(packed_embedded)  # Runs the packed sequence through the LSTM and returns outputs plus final states.

        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output)  # Converts packed LSTM outputs back to padded tensor form.

        context, attn_weights = self.attention(output)  # Applies attention to turn token-level outputs into one context vector per example.

        return self.fc(context), attn_weights  # Returns class logits and attention weights for explainability.


INPUT_DIM = padding_value  # Sets the input dimension for the LSTM model.
EMBEDDING_DIM = 100  # Sets the embedding dimension for the LSTM model.
HIDDEN_DIM = 256  # Sets the LSTM hidden-state dimension per direction.
OUTPUT_DIM = 3  # Sets the number of output sentiment classes.
N_LAYERS = 2  # Uses two stacked LSTM layers.
BIDIRECTIONAL = True  # Enables a bidirectional LSTM.
DROPOUT = 0.5  # Sets dropout probability to 50 percent.

lstm_model = LSTM(  # Instantiates the LSTM attention classifier.
    INPUT_DIM,  # Passes the input vocabulary dimension.
    EMBEDDING_DIM,  # Passes the embedding vector dimension.
    HIDDEN_DIM,  # Passes the hidden-state dimension.
    OUTPUT_DIM,  # Passes the number of output classes.
    N_LAYERS,  # Passes the number of LSTM layers.
    BIDIRECTIONAL,  # Passes whether the LSTM is bidirectional.
    DROPOUT,  # Passes the dropout probability.
    embedding_weights,  # Passes pretrained Word2Vec embedding weights.
)  # Ends LSTM model instantiation.
rnn_model = rnn_model.to(device)  # Moves the RNN model to the selected device.
lstm_model = lstm_model.to(device)  # Moves the LSTM model to the selected device.
lstm_optimizer = torch.optim.Adam(lstm_model.parameters())  # Creates an Adam optimizer for the LSTM model.
lstm_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)  # Creates weighted cross-entropy loss for LSTM training.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Reconfirms the selected compute device.



def accuracy(preds, y):  # Defines a helper function to compute batch accuracy.
    predicted = torch.argmax(preds, dim=1)  # Converts class logits into predicted class indices.
    correct = (predicted == y).float()  # Creates a float tensor where correct predictions are 1 and incorrect ones are 0.
    return correct.sum() / correct.numel()  # Returns the fraction of correct predictions in the batch.

def confidence_penalty(logits):  # Defines an entropy-based penalty to discourage overconfident predictions.
    probs = torch.softmax(logits, dim=1)  # Converts logits into class probabilities.
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # Computes entropy per sample with numerical stabilization.
    return -entropy.mean()  # Returns negative mean entropy so adding it penalizes low-entropy confidence.

def train(model, iterator, optimizer, criterion):  # Defines one full training pass over a prepared iterator.
    epoch_loss = 0  # Initializes the accumulated training loss.
    epoch_acc = 0  # Initializes the accumulated training accuracy.

    print("🟢 Entered train()")  # Prints a debug message when training starts.

    model.train()  # Puts the model into training mode so training-time behavior is enabled.

    for i, batch in enumerate(iterator):  # Iterates through mini-batches with their batch index.

        # 👉 Print ONLY first batch (important)  # Explains that the next debug block runs only once per epoch.
        if i == 0:  # Checks whether this is the first batch.
            print("🔍 First batch debug:")  # Prints a heading for first-batch diagnostics.
            print("Text shape:", batch["text"].shape)  # Prints the padded text tensor shape.
            print("Batch stats:",  # Starts printing basic statistics about token indices in the batch.
      "min =", batch["text"].min().item(),  # Prints the minimum token index in the batch.
      "max =", batch["text"].max().item(),  # Prints the maximum token index in the batch.
      "mean =", batch["text"].float().mean().item())  # Prints the mean token index after converting to float.
            print("Length shape:", batch["length"].shape)  # Prints the shape of the sequence-length tensor.
            print("Label shape:", batch["label"].shape)  # Prints the shape of the label tensor.
            

        optimizer.zero_grad()  # Clears old gradients before the new backward pass.

        predictions, attn_weights = model(batch["text"], batch["length"])  # Runs the model forward on text and sequence lengths.

        # 👉 Check predictions  # Explains that the next debug block inspects output logits.
        if i == 0:  # Checks whether this is the first batch.
            print("Predictions shape:", predictions.shape)  # Prints the shape of the logits tensor.
            print("Predictions sample:", predictions[:5])  # Prints the first five prediction rows for debugging.
            
        base_loss = criterion(predictions, batch["label"].long())  # Computes weighted cross-entropy classification loss.
        confidence_loss = confidence_penalty(predictions)  # Computes the entropy-based confidence penalty.

        loss = base_loss + 0.05 * confidence_loss  # Combines classification loss with a small confidence penalty.
        
        acc = accuracy(predictions, batch["label"].long())  # Computes batch accuracy from logits and labels.


        # 👉 Check loss  # Explains that the next debug block prints the first loss value.
        if i == 0:  # Checks whether this is the first batch.
            print("Loss value:", loss.item())  # Prints the scalar loss value for debugging.


        loss.backward()  # Backpropagates gradients through the model.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clips gradients to reduce exploding-gradient risk.

        # 👉 Check gradients  # Explains that the next debug block inspects gradient magnitudes.
        if i == 0:  # Checks whether this is the first batch.
            for name, param in model.named_parameters():  # Iterates over model parameters with their names.
                if param.grad is not None:  # Finds the first parameter that received a gradient.
                    print(f"Gradient check ({name}):", param.grad.abs().mean().item())  # Prints the mean absolute gradient for that parameter.
                    break  # only print one  # Stops after one gradient diagnostic to avoid excessive output.

        optimizer.step()  # Updates model parameters using the computed gradients.

        epoch_loss += loss.item()  # Adds the batch loss to the epoch loss accumulator.
        epoch_acc += acc.item()  # Adds the batch accuracy to the epoch accuracy accumulator.

    print("✅ Finished train()")  # Prints a status message when training for the epoch is complete.
    tracked_attention = []  # Initializes an empty attention list for API consistency, though training does not populate it.
    return epoch_loss / len(iterator), epoch_acc / len(iterator)  # Returns average training loss and accuracy.
    

def evaluate(model, iterator, criterion):  # Defines evaluation over validation or test batches.
    misclassified_examples = []  # Stores examples the model predicts incorrectly.
    correct_examples = []  # Stores examples the model predicts correctly.
    epoch_loss = 0  # Initializes accumulated evaluation loss.
    epoch_acc = 0  # Initializes accumulated evaluation accuracy.

    all_preds = []  # Stores all predicted class indices for aggregate metrics.
    all_labels = []  # Stores all true class indices for aggregate metrics.
    all_probs = []  # Stores all predicted probabilities for ROC curves.
    tracked_attention = []  # Stores a few attention vectors for later inspection.

    model.eval()  # Puts the model in evaluation mode.

    with torch.no_grad():  # Disables gradient tracking during evaluation for speed and memory savings.
        for batch in iterator:  # Iterates through evaluation mini-batches.
            predictions, attn_weights = model(batch["text"], batch["length"])  # Runs the model forward on the current batch.
            if attn_weights is not None and len(tracked_attention) < 3:  # Keeps only a few attention examples if the model provides them.
                tracked_attention.append(attn_weights[0].detach().cpu().numpy())  # Stores the first example's attention weights from this batch.
            # Convert logits → probabilities  # Explains that logits are converted before confidence and ROC calculations.
            probs = torch.softmax(predictions, dim=1)  # Converts logits into probabilities over sentiment classes.
            all_probs.extend(probs.cpu().numpy())  # Stores probabilities on CPU as NumPy arrays.

            loss = criterion(predictions, batch["label"].long())  # Computes evaluation loss for the batch.
            acc = accuracy(predictions, batch["label"].long())  # Computes evaluation accuracy for the batch.

            epoch_loss += loss.item()  # Adds batch loss to the evaluation loss accumulator.
            epoch_acc += acc.item()  # Adds batch accuracy to the evaluation accuracy accumulator.

            preds = torch.argmax(predictions, dim=1)  # Converts logits into predicted class indices.
            
            for i in range(len(preds)):  # Iterates over examples in the current batch.
                true_label = batch["label"][i].item()  # Reads the true class label for this example.
                pred_label = preds[i].item()  # Reads the predicted class label for this example.
                confidence = probs[i][pred_label].item()  # Reads the predicted probability for the chosen class.

                if true_label != pred_label:  # Checks whether the prediction is wrong.
                   misclassified_examples.append({  # Adds a record for this misclassified example.
                        "text": batch["raw_text"][i],  # Stores the original token list for display and attention plotting.
                        "true": true_label,  # Stores the true class index.
                        "pred": pred_label,  # Stores the predicted class index.
                        "confidence": confidence,  # Stores model confidence in the wrong prediction.
                        "attention": attn_weights[i].detach().cpu().numpy() if attn_weights is not None else None  # Stores attention weights if available.
                    })  # Ends the misclassified-example dictionary append.
                else:  # Handles correctly predicted examples.
                    correct_examples.append({  # Adds a record for this correctly classified example.
                       "text": batch["raw_text"][i],  # Stores the original token list for display and attention plotting.
                       "true": true_label,  # Stores the true class index.
                       "pred": pred_label,  # Stores the predicted class index.
                       "confidence": confidence,  # Stores confidence in the correct prediction.
                       "attention": attn_weights[i].detach().cpu().numpy() if attn_weights is not None else None  # Stores attention weights if available.
                    })  # Ends the correct-example dictionary append.

                
            
            all_preds.extend(preds.cpu().numpy())  # Adds current-batch predictions to the aggregate prediction list.
            all_labels.extend(batch["label"].cpu().numpy())  # Adds current-batch labels to the aggregate label list.

    # =========================  # Starts the visual evaluation section divider.
    # 📊 CONFUSION MATRIX  # Labels the confusion-matrix section.
    # =========================  # Ends the visual evaluation section divider.
    from collections import Counter  # Imports Counter locally to summarize label distributions.

    print("\n📊 True label distribution:")  # Prints a heading for true-label counts.
    print(Counter(all_labels))  # Prints counts of true labels across the evaluated split.

    print("\n📊 Predicted label distribution:")  # Prints a heading for predicted-label counts.
    print(Counter(all_preds))  # Prints counts of predicted labels across the evaluated split.
    
    cm = confusion_matrix(all_labels, all_preds)  # Computes the confusion matrix from true and predicted labels.
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # Normalizes each row by the number of actual examples in that class.
    print("\n📊 Confusion Matrix (Heatmap):")  # Prints a heading before plotting the confusion matrix.
    labels = ["negative", "neutral", "positive"]  # Defines class-name labels for reports and plots.
    
    print("\n🧠 Per-class performance (Recall / Sensitivity):")  # Prints a heading for per-class recall.
    for i, label in enumerate(labels):  # Iterates over each class name and index.
        correct = cm[i, i]  # Reads the diagonal count for correctly predicted examples of this class.
        total = cm[i].sum()  # Counts all true examples of this class.
        recall = correct / total if total > 0 else 0  # Computes class recall while avoiding division by zero.

        print(f"{label}:")  # Prints the class name.
        print(f"   ✔ Correct: {correct}/{total}")  # Prints the number of correct examples out of total actual examples.
        print(f"   📈 Recall (Sensitivity): {recall:.2%}")  # Prints recall as a percentage.
        
        print("\n⚠️ Most confused class pairs:")  # Prints a heading for off-diagonal confusion counts.

        for i in range(len(labels)):  # Iterates over actual class indices for confusion-pair reporting.
            for j in range(len(labels)):  # Iterates over predicted class indices for confusion-pair reporting.
                if i != j and cm[i, j] > 0:  # Selects only nonzero mistakes between different classes.
                    print(f"{labels[i]} → {labels[j]}: {cm[i,j]} mistakes")  # Prints one observed mistake direction and count.
        
        print("\n📊 Normalized confusion matrix (row = actual):")  # Prints a heading for normalized row values.
        for i, label in enumerate(labels):  # Iterates over class names and row indices.
            print(f"{label}: {cm_normalized[i]}")  # Prints the normalized confusion row for the class.
    
    plt.figure(figsize=(6, 5))  # Creates a Matplotlib figure for the confusion-matrix heatmap.
    sns.heatmap(  # Draws a Seaborn heatmap of the normalized confusion matrix.
        cm_normalized,  # Supplies normalized confusion-matrix values.
        annot=True,              # show numbers inside cells  # Writes numeric values into heatmap cells.
        fmt=".2f",                 # integer format  # Formats annotations with two decimal places.
        cmap="Blues",            # color style  # Uses the blue color map for the heatmap.
        xticklabels=labels,  # Sets predicted-class labels on the x-axis.
        yticklabels=labels  # Sets actual-class labels on the y-axis.
    )  # Ends heatmap creation.
    
    plt.xlabel("Predicted")  # Labels the x-axis as predicted classes.
    plt.ylabel("Actual")  # Labels the y-axis as actual classes.
    plt.title("Confusion Matrix Heatmap")  # Adds a title to the confusion-matrix plot.

    plt.tight_layout()  # Adjusts plot spacing to reduce clipping.
    plt.show()  # Displays the confusion-matrix heatmap.
    # =========================  # Starts the classification-report section divider.
    # 📈 CLASSIFICATION REPORT  # Labels the classification-report section.
    # =========================  # Ends the classification-report section divider.
    print("\n📈 Classification Report:")  # Prints a heading for precision, recall, F1, and support.
    print(classification_report(all_labels, all_preds))  # Prints scikit-learn's full classification report.

    # =========================  # Starts the F1-score section divider.
    # 🔥 F1 SCORE (MACRO)  # Labels the macro-F1 section.
    # =========================  # Ends the F1-score section divider.
    f1 = f1_score(all_labels, all_preds, average="macro")  # Computes macro-averaged F1 across classes.
    print(f"\n🔥 F1 Score (macro): {f1:.4f}")  # Prints the macro-F1 score to four decimals.



    # =========================  # Starts the multiclass ROC section divider.
    # 🎯 ROC CURVE (MULTICLASS)  # Labels the multiclass ROC section.
    # =========================  # Ends the multiclass ROC section divider.
    all_labels_bin = label_binarize(all_labels, classes=[0, 1, 2])  # Converts integer labels into one-vs-rest binary label columns.
    all_probs_np = np.array(all_probs)  # Converts stored probability rows into a NumPy array.

    fpr = dict()  # Initializes a dictionary for false-positive-rate arrays by class.
    tpr = dict()  # Initializes a dictionary for true-positive-rate arrays by class.
    roc_auc = dict()  # Initializes a dictionary for AUC values by class.

    for class_id in range(3):  # Iterates over each sentiment class for one-vs-rest ROC calculation.
        fpr[class_id], tpr[class_id], _ = roc_curve(  # Computes the ROC curve for the current class.
            all_labels_bin[:, class_id],  # Supplies binary true labels for the current one-vs-rest class.
            all_probs_np[:, class_id]  # Supplies predicted probabilities for the current class.
        )  # Ends the ROC-curve call.
        roc_auc[class_id] = auc(fpr[class_id], tpr[class_id])  # Computes the area under the ROC curve for the class.

    plt.figure()  # Creates a new figure for ROC curves.
    for class_id in range(3):  # Iterates over classes to plot each ROC curve.
        plt.plot(  # Adds one class-specific ROC curve to the plot.
            fpr[class_id],  # Supplies false-positive rates on the x-axis.
            tpr[class_id],  # Supplies true-positive rates on the y-axis.
            label=f"{labels[class_id]} (AUC = {roc_auc[class_id]:.2f})"  # Labels the curve with class name and AUC.
        )  # Ends the class-specific ROC plot call.

    plt.plot([0, 1], [0, 1], "k--")  # Draws the diagonal random-classifier baseline.
    plt.xlabel("False Positive Rate")  # Labels the x-axis.
    plt.ylabel("True Positive Rate")  # Labels the y-axis.
    plt.title("ROC Curve (Multiclass One-vs-Rest)")  # Adds a title to the ROC plot.
    plt.legend()  # Shows the legend with class names and AUC values.
    plt.show()  # Displays the ROC plot.
    
    
    print("\n🔥 HARDEST MISCLASSIFIED EXAMPLES:")  # Prints a heading for the most confident wrong predictions.
    
    def plot_attention_heatmap(words, attention_weights):  # Defines a helper to visualize attention weights over tokens.
        

        # Make sure lengths match  # Explains why attention is truncated to the token count.
        attention_weights = attention_weights[:len(words)]  # Trims attention values so they align with the displayed words.

        plt.figure(figsize=(len(words) * 0.6, 2))  # Creates a heatmap figure sized according to sentence length.

        # Convert to 2D for heatmap  # Explains that Seaborn heatmap expects a 2D array.
        weights = np.array(attention_weights).reshape(1, -1)  # Reshapes attention weights into one heatmap row.

        sns.heatmap(  # Draws the attention heatmap.
            weights,  # Supplies the 2D attention array.
            annot=True,  # Prints attention values inside cells.
            fmt=".2f",  # Formats attention values to two decimals.
            cmap="Reds",  # Uses a red color map for attention intensity.
            xticklabels=words,  # Uses tokens as x-axis labels.
            yticklabels=["Attention"]  # Uses a single y-axis label for the attention row.
       )  # Ends the attention heatmap call.

        plt.xticks(rotation=45)  # Rotates token labels for readability.
        plt.title("Attention Heatmap")  # Adds a title to the attention heatmap.
        plt.tight_layout()  # Adjusts spacing to reduce clipping.
        plt.show()  # Displays the attention heatmap.
     
    # Sort by confidence (most confident mistakes = worst errors)  # Explains the ranking criterion for errors.
    misclassified_examples = sorted(  # Sorts misclassified examples by confidence.
        misclassified_examples,  # Supplies the list of wrong predictions.
        key=lambda x: x["confidence"],  # Sorts using prediction confidence.
        reverse=True  # Places the highest-confidence mistakes first.
    )  # Ends sorting of misclassified examples.

    label_names = ["negative", "neutral", "positive"]  # Defines readable class names for printed examples.

    for i, example in enumerate(misclassified_examples[:10]):  # Iterates through the top ten hardest mistakes.
        print(f"\nExample {i+1}")  # Prints the example number.
        print(f"Text: {example['text']}")  # Prints the tokenized text for the example.
        print(f"True: {label_names[example['true']]}")  # Prints the true sentiment name.
        print(f"Predicted: {label_names[example['pred']]}")  # Prints the predicted sentiment name.
        print(f"Confidence: {example['confidence']:.2f}")  # Prints prediction confidence to two decimals.
        
        words = example["text"]  # Stores tokens for heatmap x-axis labels.
        attention = example["attention"]  # Stores attention weights for plotting.

        if attention is not None:  # Checks whether this model produced attention weights.
            plot_attention_heatmap(words, attention)  # Plots attention over tokens for the example.
        else:  # Handles models without attention, such as the vanilla RNN.
            print("No attention weights available for this model.")  # Prints that no attention visualization can be shown.
       
    print("\n✅ CORRECT PREDICTIONS:")  # Prints a heading for confident correct examples.

    correct_examples = sorted(  # Sorts correctly predicted examples by confidence.
        correct_examples,  # Supplies the list of correct predictions.
        key=lambda x: x["confidence"],  # Sorts using prediction confidence.
        reverse=True  # Places the highest-confidence correct examples first.
    )  # Ends sorting of correct examples.

    for i, example in enumerate(correct_examples[:5]):  # Iterates through the top five confident correct examples.

        print(f"\nExample {i+1}")  # Prints the example number.
        print(f"Text: {example['text']}")  # Prints the tokenized text for the example.
        print(f"True: {label_names[example['true']]}")  # Prints the true sentiment name.
        print(f"Confidence: {example['confidence']:.2f}")  # Prints confidence in the correct prediction.

        # 🔥 ADD HEATMAP HERE  # Marks where attention visualization is added for correct examples.
        words = example["text"]  # Stores tokens for heatmap x-axis labels.
        attention = example["attention"]  # Stores attention weights for plotting.

        if attention is not None:  # Checks whether attention weights exist for this example.
            plot_attention_heatmap(words, attention)  # Plots the attention heatmap for the correct example.
        else:  # Handles models without attention weights.
            print("No attention weights available for this model.")  # Prints that no attention visualization is available.

    
    return epoch_loss / len(iterator), epoch_acc / len(iterator), tracked_attention  # Returns average loss, average accuracy, and stored attention examples.
    
    

batch_size = 32  # Usually should be a power of 2 because it's the easiest for computer memory.


def iterator(X, y, raw_texts=None, shuffle=True):  # Defines a custom mini-batch iterator for variable-length token-index tensors.
    size = len(X)  # Stores the number of examples in the split.
    if shuffle:  # Checks whether the examples should be shuffled.
        permutation = np.random.permutation(size)  # Creates a random order of example indices.
    else:  # Handles deterministic evaluation order.
        permutation = np.arange(size)  # Creates a sequential order of example indices.
    iterate = []  # Initializes the list that will store prepared mini-batches.
    for i in range(0, size, batch_size):  # Steps through the dataset in chunks of batch_size.
        indices = permutation[i : i + batch_size]  # Selects the indices for the current mini-batch.
        batch = {}  # Initializes a dictionary to hold tensors and metadata for the batch.

        valid_indices = [idx for idx in indices if len(X[idx]) > 0]  # Removes examples whose token-index sequence is empty.

        batch["text"] = [X[idx] for idx in valid_indices]  # Collects token-index tensors for valid examples.
        batch["label"] = [y[idx] for idx in valid_indices]  # Collects labels for valid examples.

        if raw_texts is not None:  # Checks whether raw/token text was supplied for explainability output.
            batch["raw_text"] = [raw_texts[idx] for idx in valid_indices]  # Collects corresponding raw/token text entries.
        else:  # Handles cases where no raw text is supplied.
            batch["raw_text"] = [None for _ in valid_indices]  # Fills raw text with None values.

        if len(batch["text"]) == 0:  # Checks whether every item in the batch was filtered out.
            continue  # Skips empty batches.

        batch["text"], batch["label"], batch["raw_text"] = zip(  # Reassigns sorted text, label, and raw text lists.
            *sorted(  # Sorts examples together while preserving alignment across fields.
                zip(batch["text"], batch["label"], batch["raw_text"]),  # Zips text, labels, and raw text into aligned triples.
                key=lambda x: len(x[0]),  # Sorts by sequence length.
                reverse=True,  # Sorts longest sequences first.
            )  # Ends sorting of aligned triples.
   )  # Ends unzip assignment after sorting.
        batch["length"] = [len(utt) for utt in batch["text"]]  # Computes true sequence lengths before padding.
        batch["length"] = torch.IntTensor(batch["length"])  # Converts sequence lengths to a PyTorch integer tensor.
        batch["text"] = torch.nn.utils.rnn.pad_sequence(  # Pads variable-length token-index tensors to a common length.
            batch["text"], batch_first=True  # Pads with batch dimension first before transposing below.
        ).t()  # Transposes padded text to sequence-first shape expected by default RNN/LSTM layers.
        batch["label"] = torch.tensor(batch["label"], dtype=torch.long)  # Converts labels to a long integer tensor for cross-entropy loss.

        batch["label"] = batch["label"].to(device)  # Moves labels to the selected compute device.
        batch["length"] = batch["length"].to(device)  # Moves sequence lengths to the selected compute device.
        batch["text"] = batch["text"].to(device)  # Moves padded token-index text tensors to the selected compute device.

        iterate.append(batch)  # Appends the prepared batch dictionary to the iterator list.

    return iterate  # Returns all prepared mini-batches.

print("🔁 Converting text to indices...")  # Prints a status message before mapping tokens to vocabulary indices.
index_utt = [  # Builds one tensor of token indices for each tokenized utterance.
    torch.tensor([word_vectors.key_to_index.get(word, 0) for word in text])  # Maps each word to its Word2Vec index, defaulting unknowns to 0.
    for text in text_data  # Iterates over every lemmatized token list.
]  # Ends the indexed utterance list.
print("Sample indexed sentence:", index_utt[0])  # Prints the first indexed sentence tensor.

raw_text_utt = text_data  # Stores the lemmatized token lists as raw text for later display and attention labels.
print("Sample raw sentence:", raw_text_utt[0])  # Prints the first lemmatized token list.

# You've got to determine some labels for whatever you're training on.  # Notes that supervised training requires labels.
print("📊 Splitting dataset...")  # Prints a status message before splitting data.
X_train, X_test, raw_train, raw_test, y_train, y_test = train_test_split(  # Splits indexed data, raw text, and labels into train and test sets.
    index_utt,  # Supplies indexed token sequences.
    raw_text_utt,  # Supplies raw/token text aligned with indexed sequences.
    label_data,  # Supplies numeric sentiment labels.
    test_size=0.2,  # Reserves 20 percent of data for testing.
    random_state=42  # Uses a fixed random seed for reproducibility.
)  # Ends the train/test split call.
print("Train size:", len(X_train))  # Prints the number of training examples before validation split.
print("Test size:", len(X_test))  # Prints the number of test examples.

print("Splitting train set into train and validation sets...")  # Prints a status message before creating validation data.
X_train, X_val, raw_train, raw_val, y_train, y_val = train_test_split(  # Splits the training set into smaller train and validation sets.
    X_train,  # Supplies current training indexed sequences.
    raw_train,  # Supplies current training raw/token text.
    y_train,  # Supplies current training labels.
    test_size=0.2,  # Reserves 20 percent of the training set for validation.
    random_state=42  # Uses the same fixed random seed for reproducibility.
)  # Ends the train/validation split call.
print("Train size:", len(X_train))  # Prints the final training-set size.
print("Validation size:", len(X_val))  # Prints the validation-set size.

print("📦 Creating validation and test iterators...")  # Prints a status message before batching validation and test data.

validate_iterator = iterator(X_val, y_val, raw_val, shuffle=False)  # Creates deterministic validation batches.
test_iterator = iterator(X_test, y_test, raw_test, shuffle=False)  # Creates deterministic test batches.

print("Validation batches:", len(validate_iterator))  # Prints the number of validation batches.
print("Test batches:", len(test_iterator))  # Prints the number of test batches.

print(len(validate_iterator), len(test_iterator))  # Prints validation and test batch counts together.


N_EPOCHS = 25  # Sets the maximum number of training epochs per model.



print("🚀 Starting training loop...")  # Prints a status message before model training begins.

for model in [rnn_model, lstm_model]:  # Trains and evaluates the RNN first, then the LSTM.

    print("===================================================")  # Prints a visual separator between model runs.
    print(f"Training {model.__class__.__name__}")  # Prints the current model class name.
    
    train_losses = []  # Stores training loss values across epochs.
    valid_losses = []  # Stores validation loss values across epochs.

    train_accuracies = []  # Stores training accuracy values across epochs.
    valid_accuracies = []  # Stores validation accuracy values across epochs.

    attention_history = []  # Stores tracked attention examples across epochs.

    best_valid_loss = float('inf')  # Initializes the best validation loss as infinity.
    best_epoch = 0  # Stores the epoch index with the best validation loss.
    patience = 3  # Sets early stopping patience to three non-improving epochs.
    patience_counter = 0  # Counts consecutive epochs without validation improvement.

    overfitting_detected = False  # Tracks whether the hard-stop overfitting rule is triggered.
    overfitting_epoch = None  # Stores the epoch where overfitting is detected.

    for epoch in range(N_EPOCHS):  # Iterates through the maximum epoch count.

        print(f"\n🔁 Epoch {epoch+1}")  # Prints the human-readable epoch number.
        
        
        # Recreate training batches every epoch  # Explains why the training iterator is rebuilt.
        train_iterator = iterator(X_train, y_train, raw_train, shuffle=True)  # Creates shuffled training batches for this epoch.

        # TRAIN  # Marks the training phase.
        if isinstance(model, RNN):  # Checks whether the current model is the vanilla RNN.
            optimizer = rnn_optimizer  # Uses the RNN optimizer.
            criterion = rnn_criterion  # Uses the RNN loss criterion.
        else:  # Handles the LSTM model.
            optimizer = lstm_optimizer  # Uses the LSTM optimizer.
            criterion = lstm_criterion  # Uses the LSTM loss criterion.

        train_loss, train_acc = train(  # Runs one training epoch and receives average metrics.
            model, train_iterator, optimizer, criterion  # Supplies model, training batches, optimizer, and loss function.
       )  # Ends the training call.

        # VALIDATE  # Marks the validation phase.
        valid_loss, valid_acc, tracked_attention = evaluate(  # Evaluates the model on validation batches.
            model,  # Supplies the current model.
            validate_iterator,  # Supplies validation batches.
            criterion  # Supplies the loss criterion.
     )  # Ends validation evaluation.

        # STORE LOSSES  # Marks metric storage for losses.
        train_losses.append(train_loss)  # Saves this epoch's training loss.
        valid_losses.append(valid_loss)  # Saves this epoch's validation loss.
        
        # =========================  # Starts the hard-stop overfitting section divider.
# =========================  # Keeps the original repeated divider comment.
# 🔥 HARD STOP OVERFITTING  # Labels the hard-stop overfitting section.
# =========================  # Ends the hard-stop overfitting section divider.

        if len(valid_losses) > 3:  # Starts hard-stop checking only after more than three validation losses exist.
            recent = valid_losses[-3:]  # Takes the three most recent validation losses.

            if recent[2] > recent[1] and recent[1] > recent[0]:  # Detects three validation losses increasing in a row.
                print("❌ Overfitting detected")  # Prints an overfitting warning.

                overfitting_detected = True  # Records that overfitting was detected.
                overfitting_epoch = epoch  # Records the current epoch index as the overfitting epoch.

                break  # Stops the epoch loop immediately after detecting overfitting.
            else:  # Handles the case where validation loss is not strictly worsening three times.
                print("✅ Validation improving")  # Prints that the hard-stop rule did not detect overfitting.
        
        # 🔥 STORE ACCURACY  # Marks metric storage for accuracies.
        train_accuracies.append(train_acc)  # Saves this epoch's training accuracy.
        valid_accuracies.append(valid_acc)  # Saves this epoch's validation accuracy.

        # STORE ATTENTION  # Marks storage for attention snapshots.
        attention_history.append(tracked_attention)  # Saves tracked validation attention examples from this epoch.

        print(f"Train Loss: {train_loss:.3f}")  # Prints training loss rounded to three decimals.
        print(f"Valid Loss: {valid_loss:.3f}")  # Prints validation loss rounded to three decimals.

        # =========================  # Starts the early-stopping section divider.
        # 🔥 EARLY STOPPING LOGIC  # Labels the early-stopping section.
        # =========================  # Ends the early-stopping section divider.

        
        if valid_loss < best_valid_loss:  # Checks whether validation loss improved.
            best_valid_loss = valid_loss  # Updates the best validation loss.
            best_epoch = epoch  # Records the best epoch index.
            patience_counter = 0  # Resets patience because the model improved.

            best_model_state = copy.deepcopy(model.state_dict())  # Deep-copies the best model weights so later updates cannot mutate them.

            torch.save(best_model_state, f"best_{model.__class__.__name__}.pth")  # Saves the best model weights to disk.

            print("✅ New best model found and saved!")  # Prints a success message for the improved model.

        else:  # Handles an epoch without validation-loss improvement.
            patience_counter += 1  # Increments the no-improvement counter.
            print(f"⚠️ No improvement ({patience_counter}/{patience})")  # Prints the early-stopping patience status.

        if patience_counter >= patience:  # Checks whether patience has been exhausted.
            print("🛑 Early stopping triggered!")  # Prints an early-stopping message.
            break  # Exits the epoch loop.

    # Restore best model  # Explains that evaluation uses the best saved in-memory weights.
    model.load_state_dict(best_model_state)  # Restores the model weights from the best validation epoch.
    print("\n🧪 FINAL TEST SET EVALUATION")  # Prints a heading before final test evaluation.
    test_loss, test_acc, test_attention = evaluate(  # Evaluates the best model on the held-out test set.
    model,  # Supplies the restored model.
    test_iterator,  # Supplies test batches.
    criterion  # Supplies the matching loss criterion.
   )  # Ends the test evaluation call.

    print(f"Test Loss: {test_loss:.4f}")  # Prints final test loss to four decimals.
    print(f"Test Accuracy: {test_acc*100:.2f}%")  # Prints final test accuracy as a percentage.
    print(f"\n🏆 Best epoch: {best_epoch+1}")  # Prints the one-based best epoch number.
    
    plt.figure()  # Creates a new figure for the loss curves.

    plt.plot(train_losses, label="Train Loss")  # Plots training loss across epochs.
    plt.plot(valid_losses, label="Validation Loss")  # Plots validation loss across epochs.

    plt.xlabel("Epoch")  # Labels the x-axis as epochs.
    plt.ylabel("Loss")  # Labels the y-axis as loss.
    plt.title("Training vs Validation Loss")  # Adds a title to the loss plot.

    plt.legend()  # Shows the legend for train and validation loss.
    plt.show()  # Displays the loss plot.
    
    print("\n🧠 ATTENTION EVOLUTION")  # Prints a heading before showing stored attention snapshots.

    for epoch_idx in range(min(5, len(attention_history))):  # Iterates over up to the first five epochs of attention history.

        print(f"\nEpoch {epoch_idx+1}")  # Prints the one-based epoch number for attention output.

        for ex_idx, attn in enumerate(attention_history[epoch_idx]):  # Iterates through tracked attention examples for that epoch.

            print(f"Example {ex_idx+1}: {attn[:10]}")  # Prints the first ten attention weights for the example.
            
    plt.figure()  # Creates a new figure for the accuracy curves.

    plt.plot(train_accuracies, label="Train Accuracy")  # Plots training accuracy across epochs.
    plt.plot(valid_accuracies, label="Validation Accuracy")  # Plots validation accuracy across epochs.

    plt.xlabel("Epoch")  # Labels the x-axis as epochs.
    plt.ylabel("Accuracy")  # Labels the y-axis as accuracy.
    plt.title("Training vs Validation Accuracy")  # Adds a title to the accuracy plot.

    plt.legend()  # Shows the legend for train and validation accuracy.
    plt.show()  # Displays the accuracy plot.
    
    print(f"\n🏆 Best epoch: {best_epoch+1}")  # Prints the one-based best epoch number again.
    print(f"Best validation loss: {best_valid_loss:.4f}")  # Prints the best validation loss.
    print(f"Validation accuracy at best epoch: {valid_accuracies[best_epoch]:.4f}")  # Prints validation accuracy at the best epoch index.
    
    print("\n🧠 OVERFITTING ANALYSIS")  # Prints a heading for the final overfitting summary.

    if overfitting_detected:  # Checks whether the hard-stop rule detected overfitting.
        print(f"❌ Overfitting started at epoch {overfitting_epoch+1}")  # Prints the one-based overfitting epoch.
    else:  # Handles the case where overfitting was not detected by the rule.
        print("✅ No overfitting detected")  # Prints a no-overfitting summary.
