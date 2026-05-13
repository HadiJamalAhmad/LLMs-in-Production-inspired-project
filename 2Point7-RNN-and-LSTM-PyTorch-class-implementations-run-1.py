# %%
#!/usr/bin/env python
# coding: utf-8

# In[5]:


import sys
print(sys.executable)

import copy
import torch
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import nltk
import spacy
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
try:
    tokenizer = nltk.tokenize.RegexpTokenizer("\w+'?\w+|\w+'")
    tokenizer.tokenize("This is a test")
    stop_words = nltk.corpus.stopwords.words("english")
    nlp = spacy.load("en_core_web_lg", disable=["parser", "ner"])
except Exception:
    nltk.download("stopwords")
    nltk.download("punkt")
    spacy.cli.download("en_core_web_lg")
    tokenizer = nltk.tokenize.RegexpTokenizer("\w+'?\w+|\w+'")
    tokenizer.tokenize("This is a test")
    stop_words = nltk.corpus.stopwords.words("english")
    nlp = spacy.load("en_core_web_lg", disable=["parser", "ner"])

# Create our corpus for training and perform some classic NLP preprocessing
print("✅ Loading dataset...")
dataset = pd.read_csv(r"C:\Users\HADI\Downloads\twitter.csv")
dataset = dataset.dropna(subset=["text"])
print("Dataset shape:", dataset.shape)
print("Columns:", dataset.columns)
print(dataset.head())

print("🔄 Tokenizing text...")
text_data = list(
    map(
        lambda x: [
            word for word in tokenizer.tokenize(x.lower())
            if word not in stop_words
        ],
        dataset["text"],
    )
)
print("Sample tokenized text:", text_data[:2])

print("🧠 Lemmatizing text (this may take time)...")
docs = list(nlp.pipe([" ".join(text) for text in text_data], batch_size=256))
text_data = [[token.lemma_ for token in doc] for doc in docs]

print("sample Lemmatized text:", text_data[:2])


print("🏷️ Processing labels...")

# Remove empty token lists and keep labels aligned
filtered_pairs = [
    (tokens, label)
    for tokens, label in zip(text_data, dataset["sentiment"])
    if len(tokens) > 0
]

text_data = [tokens for tokens, _ in filtered_pairs]

label_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}
label_data = [label_map[label] for _, label in filtered_pairs]

print("Sample labels:", label_data[:10])
print("Number of samples:", len(label_data))

from collections import Counter

print("\n📊 Class distribution:")
class_counts = Counter(label_data)
print(class_counts)

num_classes = 3
total_samples = len(label_data)

class_weights = []
for class_id in range(num_classes):
    class_count = class_counts[class_id]
    weight = total_samples / (num_classes * class_count)
    class_weights.append(weight)

class_weights = torch.tensor(class_weights, dtype=torch.float)

print("⚖️ Class weights:", class_weights)

assert len(text_data) == len(
    label_data
), f"{len(text_data)} does not equal {len(label_data)}"

EMBEDDING_DIM = 100

print("⚙️ Training Word2Vec...")
model = Word2Vec(
    text_data, vector_size=EMBEDDING_DIM, window=5, min_count=1, workers=4
)
print("✅ Word2Vec trained")
word_vectors = model.wv
print(f"Vocabulary Length: {len(model.wv)}")
del model

padding_value = len(word_vectors.index_to_key)

# Embeddings are needed to give semantic value to the inputs of an LSTM
print("🔢 Creating embedding weights...")
embedding_weights = torch.Tensor(word_vectors.vectors)

# Add padding vector (zeros)
pad_vector = torch.zeros(1, embedding_weights.shape[1])
embedding_weights = torch.cat((embedding_weights, pad_vector), dim=0)

padding_value = embedding_weights.shape[0] - 1
print("Embedding shape:", embedding_weights.shape)


class KhushuAttention(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = torch.nn.Linear(hidden_dim * 2, 1)
        

    def forward(self, outputs):
        # outputs: [seq_len, batch, hidden_dim*2]
        outputs = outputs.permute(1, 0, 2)  # [batch, seq, hidden]
        
        scores = self.attn(outputs).squeeze(-1)  # [batch, seq]
        weights = torch.softmax(scores, dim=1)

        context = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)

        return context, weights


class RNN(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        output_dim,
        embedding_weights,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(
        embedding_weights,
        padding_idx=padding_value,
        freeze=True
        )   
        self.rnn = torch.nn.RNN(embedding_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, text_lengths):
        embedded = self.embedding(x)
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(
             embedded,
             text_lengths.cpu(),
             enforce_sorted=False
    )
        packed_output, hidden = self.rnn(packed_embedded)
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output
        )
        return self.fc(hidden.squeeze(0)), None


INPUT_DIM = padding_value
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 3

rnn_model = RNN(
    INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, embedding_weights
)

rnn_optimizer = torch.optim.SGD(rnn_model.parameters(), lr=1e-3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = class_weights.to(device)
rnn_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

class LSTM(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout,
        embedding_weights,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(
        embedding_weights,
        padding_idx=padding_value,
        freeze=True
        )    
        self.rnn = torch.nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.fc = torch.nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.attention = KhushuAttention(hidden_dim)
        
    def forward(self, x, text_lengths):
        embedded = self.embedding(x)
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(
    embedded, text_lengths.cpu(), enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output)

        context, attn_weights = self.attention(output)

        return self.fc(context), attn_weights


INPUT_DIM = padding_value
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 3
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

lstm_model = LSTM(
    INPUT_DIM,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    OUTPUT_DIM,
    N_LAYERS,
    BIDIRECTIONAL,
    DROPOUT,
    embedding_weights,
)
rnn_model = rnn_model.to(device)
lstm_model = lstm_model.to(device)
lstm_optimizer = torch.optim.Adam(lstm_model.parameters())
lstm_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def accuracy(preds, y):
    predicted = torch.argmax(preds, dim=1)
    correct = (predicted == y).float()
    return correct.sum() / correct.numel()

def confidence_penalty(logits):
    probs = torch.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    return -entropy.mean()

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    print("🟢 Entered train()")

    model.train()

    for i, batch in enumerate(iterator):

        # 👉 Print ONLY first batch (important)
        if i == 0:
            print("🔍 First batch debug:")
            print("Text shape:", batch["text"].shape)
            print("Batch stats:",
      "min =", batch["text"].min().item(),
      "max =", batch["text"].max().item(),
      "mean =", batch["text"].float().mean().item())
            print("Length shape:", batch["length"].shape)
            print("Label shape:", batch["label"].shape)
            

        optimizer.zero_grad()

        predictions, attn_weights = model(batch["text"], batch["length"])

        # 👉 Check predictions
        if i == 0:
            print("Predictions shape:", predictions.shape)
            print("Predictions sample:", predictions[:5])
            
        base_loss = criterion(predictions, batch["label"].long())
        confidence_loss = confidence_penalty(predictions)

        loss = base_loss + 0.05 * confidence_loss
        
        acc = accuracy(predictions, batch["label"].long())


        # 👉 Check loss
        if i == 0:
            print("Loss value:", loss.item())


        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 👉 Check gradients
        if i == 0:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"Gradient check ({name}):", param.grad.abs().mean().item())
                    break  # only print one

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    print("✅ Finished train()")
    tracked_attention = []
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
    

def evaluate(model, iterator, criterion):
    misclassified_examples = []
    correct_examples = []
    epoch_loss = 0
    epoch_acc = 0

    all_preds = []
    all_labels = []
    all_probs = []
    tracked_attention = []

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions, attn_weights = model(batch["text"], batch["length"])
            if attn_weights is not None and len(tracked_attention) < 3:
                tracked_attention.append(attn_weights[0].detach().cpu().numpy())
            # Convert logits → probabilities
            probs = torch.softmax(predictions, dim=1)
            all_probs.extend(probs.cpu().numpy())

            loss = criterion(predictions, batch["label"].long())
            acc = accuracy(predictions, batch["label"].long())

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            preds = torch.argmax(predictions, dim=1)
            
            for i in range(len(preds)):
                true_label = batch["label"][i].item()
                pred_label = preds[i].item()
                confidence = probs[i][pred_label].item()

                if true_label != pred_label:
                   misclassified_examples.append({
                        "text": batch["raw_text"][i],
                        "true": true_label,
                        "pred": pred_label,
                        "confidence": confidence,
                        "attention": attn_weights[i].detach().cpu().numpy() if attn_weights is not None else None
                    })
                else:
                    correct_examples.append({
                       "text": batch["raw_text"][i],
                       "true": true_label,
                       "pred": pred_label,
                       "confidence": confidence,
                       "attention": attn_weights[i].detach().cpu().numpy() if attn_weights is not None else None
                    })

                
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["label"].cpu().numpy())

    # =========================
    # 📊 CONFUSION MATRIX
    # =========================
    from collections import Counter

    print("\n📊 True label distribution:")
    print(Counter(all_labels))

    print("\n📊 Predicted label distribution:")
    print(Counter(all_preds))
    
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    print("\n📊 Confusion Matrix (Heatmap):")
    labels = ["negative", "neutral", "positive"]
    
    print("\n🧠 Per-class performance (Recall / Sensitivity):")
    for i, label in enumerate(labels):
        correct = cm[i, i]
        total = cm[i].sum()
        recall = correct / total if total > 0 else 0

        print(f"{label}:")
        print(f"   ✔ Correct: {correct}/{total}")
        print(f"   📈 Recall (Sensitivity): {recall:.2%}")
        
        print("\n⚠️ Most confused class pairs:")

        for i in range(len(labels)):
            for j in range(len(labels)):
                if i != j and cm[i, j] > 0:
                    print(f"{labels[i]} → {labels[j]}: {cm[i,j]} mistakes")
        
        print("\n📊 Normalized confusion matrix (row = actual):")
        for i, label in enumerate(labels):
            print(f"{label}: {cm_normalized[i]}")
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_normalized,
        annot=True,              # show numbers inside cells
        fmt=".2f",                 # integer format
        cmap="Blues",            # color style
        xticklabels=labels,
        yticklabels=labels
    )
    
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix Heatmap")

    plt.tight_layout()
    plt.show()
    # =========================
    # 📈 CLASSIFICATION REPORT
    # =========================
    print("\n📈 Classification Report:")
    print(classification_report(all_labels, all_preds))

    # =========================
    # 🔥 F1 SCORE (MACRO)
    # =========================
    f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"\n🔥 F1 Score (macro): {f1:.4f}")



    # =========================
    # 🎯 ROC CURVE (MULTICLASS)
    # =========================
    all_labels_bin = label_binarize(all_labels, classes=[0, 1, 2])
    all_probs_np = np.array(all_probs)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for class_id in range(3):
        fpr[class_id], tpr[class_id], _ = roc_curve(
            all_labels_bin[:, class_id],
            all_probs_np[:, class_id]
        )
        roc_auc[class_id] = auc(fpr[class_id], tpr[class_id])

    plt.figure()
    for class_id in range(3):
        plt.plot(
            fpr[class_id],
            tpr[class_id],
            label=f"{labels[class_id]} (AUC = {roc_auc[class_id]:.2f})"
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Multiclass One-vs-Rest)")
    plt.legend()
    plt.show()
    
    
    print("\n🔥 HARDEST MISCLASSIFIED EXAMPLES:")
    
    def plot_attention_heatmap(words, attention_weights):
        

        # Make sure lengths match
        attention_weights = attention_weights[:len(words)]

        plt.figure(figsize=(len(words) * 0.6, 2))

        # Convert to 2D for heatmap
        weights = np.array(attention_weights).reshape(1, -1)

        sns.heatmap(
            weights,
            annot=True,
            fmt=".2f",
            cmap="Reds",
            xticklabels=words,
            yticklabels=["Attention"]
       )

        plt.xticks(rotation=45)
        plt.title("Attention Heatmap")
        plt.tight_layout()
        plt.show() 
     
    # Sort by confidence (most confident mistakes = worst errors)
    misclassified_examples = sorted(
        misclassified_examples,
        key=lambda x: x["confidence"],
        reverse=True
    )

    label_names = ["negative", "neutral", "positive"]

    for i, example in enumerate(misclassified_examples[:10]):  # top 10
        print(f"\nExample {i+1}")
        print(f"Text: {example['text']}")
        print(f"True: {label_names[example['true']]}")
        print(f"Predicted: {label_names[example['pred']]}")
        print(f"Confidence: {example['confidence']:.2f}")
        
        words = example["text"]
        attention = example["attention"]

        if attention is not None:
            plot_attention_heatmap(words, attention)
        else:
            print("No attention weights available for this model.")
       
    print("\n✅ CORRECT PREDICTIONS:")

    correct_examples = sorted(
        correct_examples,
        key=lambda x: x["confidence"],
        reverse=True
    )

    for i, example in enumerate(correct_examples[:5]):

        print(f"\nExample {i+1}")
        print(f"Text: {example['text']}")
        print(f"True: {label_names[example['true']]}")
        print(f"Confidence: {example['confidence']:.2f}")

        # 🔥 ADD HEATMAP HERE
        words = example["text"]
        attention = example["attention"]

        if attention is not None:
            plot_attention_heatmap(words, attention)
        else:
            print("No attention weights available for this model.")

    
    return epoch_loss / len(iterator), epoch_acc / len(iterator), tracked_attention
    
    

batch_size = 32  # Usually should be a power of 2 because it's the easiest for computer memory.


def iterator(X, y, raw_texts=None, shuffle=True):
    size = len(X)
    if shuffle:
        permutation = np.random.permutation(size)
    else:
        permutation = np.arange(size)
    iterate = []
    for i in range(0, size, batch_size):
        indices = permutation[i : i + batch_size]
        batch = {}

        valid_indices = [idx for idx in indices if len(X[idx]) > 0]

        batch["text"] = [X[idx] for idx in valid_indices]
        batch["label"] = [y[idx] for idx in valid_indices]

        if raw_texts is not None:
            batch["raw_text"] = [raw_texts[idx] for idx in valid_indices]
        else:
            batch["raw_text"] = [None for _ in valid_indices]

        if len(batch["text"]) == 0:
            continue

        batch["text"], batch["label"], batch["raw_text"] = zip(
            *sorted(
                zip(batch["text"], batch["label"], batch["raw_text"]),
                key=lambda x: len(x[0]),
                reverse=True,
            )
   )
        batch["length"] = [len(utt) for utt in batch["text"]]
        batch["length"] = torch.IntTensor(batch["length"])
        batch["text"] = torch.nn.utils.rnn.pad_sequence(
            batch["text"], batch_first=True
        ).t()
        batch["label"] = torch.tensor(batch["label"], dtype=torch.long)

        batch["label"] = batch["label"].to(device)
        batch["length"] = batch["length"].to(device)
        batch["text"] = batch["text"].to(device)

        iterate.append(batch)

    return iterate

print("🔁 Converting text to indices...")
index_utt = [
    torch.tensor([word_vectors.key_to_index.get(word, 0) for word in text])
    for text in text_data
]
print("Sample indexed sentence:", index_utt[0])

raw_text_utt = text_data
print("Sample raw sentence:", raw_text_utt[0])

# You've got to determine some labels for whatever you're training on.
print("📊 Splitting dataset...")
X_train, X_test, raw_train, raw_test, y_train, y_test = train_test_split(
    index_utt,
    raw_text_utt,
    label_data,
    test_size=0.2,
    random_state=42
)
print("Train size:", len(X_train))
print("Test size:", len(X_test))

print("Splitting train set into train and validation sets...")
X_train, X_val, raw_train, raw_val, y_train, y_val = train_test_split(
    X_train,
    raw_train,
    y_train,
    test_size=0.2,
    random_state=42
)
print("Train size:", len(X_train))
print("Validation size:", len(X_val))

print("📦 Creating validation and test iterators...")

validate_iterator = iterator(X_val, y_val, raw_val, shuffle=False)
test_iterator = iterator(X_test, y_test, raw_test, shuffle=False)

print("Validation batches:", len(validate_iterator))
print("Test batches:", len(test_iterator))

print(len(validate_iterator), len(test_iterator))


N_EPOCHS = 25



print("🚀 Starting training loop...")

for model in [rnn_model, lstm_model]:

    print("===================================================")
    print(f"Training {model.__class__.__name__}")
    
    train_losses = []
    valid_losses = []

    train_accuracies = []
    valid_accuracies = []

    attention_history = []

    best_valid_loss = float('inf')
    best_epoch = 0
    patience = 3
    patience_counter = 0

    overfitting_detected = False
    overfitting_epoch = None

    for epoch in range(N_EPOCHS):

        print(f"\n🔁 Epoch {epoch+1}")
        
        
        # Recreate training batches every epoch
        train_iterator = iterator(X_train, y_train, raw_train, shuffle=True)

        # TRAIN
        if isinstance(model, RNN):
            optimizer = rnn_optimizer
            criterion = rnn_criterion
        else:
            optimizer = lstm_optimizer
            criterion = lstm_criterion

        train_loss, train_acc = train(
            model, train_iterator, optimizer, criterion
       )

        # VALIDATE
        valid_loss, valid_acc, tracked_attention = evaluate(
            model,
            validate_iterator,
            criterion
     )

        # STORE LOSSES
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        # =========================
# =========================
# 🔥 HARD STOP OVERFITTING
# =========================

        if len(valid_losses) > 3:
            recent = valid_losses[-3:]

            if recent[2] > recent[1] and recent[1] > recent[0]:
                print("❌ Overfitting detected")

                overfitting_detected = True
                overfitting_epoch = epoch

                break
            else:
                print("✅ Validation improving")
        
        # 🔥 STORE ACCURACY
        train_accuracies.append(train_acc)
        valid_accuracies.append(valid_acc)

        # STORE ATTENTION
        attention_history.append(tracked_attention)

        print(f"Train Loss: {train_loss:.3f}")
        print(f"Valid Loss: {valid_loss:.3f}")

        # =========================
        # 🔥 EARLY STOPPING LOGIC
        # =========================

        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            patience_counter = 0

            best_model_state = copy.deepcopy(model.state_dict())

            torch.save(best_model_state, f"best_{model.__class__.__name__}.pth")

            print("✅ New best model found and saved!")

        else:
            patience_counter += 1
            print(f"⚠️ No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print("🛑 Early stopping triggered!")
            break

    # Restore best model
    model.load_state_dict(best_model_state)
    print("\n🧪 FINAL TEST SET EVALUATION")
    test_loss, test_acc, test_attention = evaluate(
    model,
    test_iterator,
    criterion
   )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"\n🏆 Best epoch: {best_epoch+1}")
    
    plt.figure()

    plt.plot(train_losses, label="Train Loss")
    plt.plot(valid_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")

    plt.legend()
    plt.show()
    
    print("\n🧠 ATTENTION EVOLUTION")

    for epoch_idx in range(min(5, len(attention_history))):

        print(f"\nEpoch {epoch_idx+1}")

        for ex_idx, attn in enumerate(attention_history[epoch_idx]):

            print(f"Example {ex_idx+1}: {attn[:10]}")
            
    plt.figure()

    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(valid_accuracies, label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")

    plt.legend()
    plt.show()
    
    print(f"\n🏆 Best epoch: {best_epoch+1}")
    print(f"Best validation loss: {best_valid_loss:.4f}")
    print(f"Validation accuracy at best epoch: {valid_accuracies[best_epoch]:.4f}")
    
    print("\n🧠 OVERFITTING ANALYSIS")

    if overfitting_detected:
        print(f"❌ Overfitting started at epoch {overfitting_epoch+1}")
    else:
        print("✅ No overfitting detected")


