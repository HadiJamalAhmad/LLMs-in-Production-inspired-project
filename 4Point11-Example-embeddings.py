import numpy as np  # Import NumPy to store embeddings as float32 arrays.
from sentence_transformers import SentenceTransformer  # Import SentenceTransformer to create sentence embeddings.
from datasets import load_dataset  # Import load_dataset to download and load Hugging Face datasets.

# Download embedding model and dataset  # Explain that the next lines load the embedding model and dataset splits.
model_ckpt = "sentence-transformers/all-MiniLM-L6-v2"  # Store the name of the pretrained SentenceTransformer checkpoint.
model = SentenceTransformer(model_ckpt)  # Load the pretrained sentence embedding model.

embs_train = load_dataset("tweet_eval", "emoji", split="train[:1000]")  # Load the first 1000 examples from the TweetEval emoji training split.
embs_test = load_dataset("tweet_eval", "emoji", split="test[:100]")  # Load the first 100 examples from the TweetEval emoji test split.


# Create embeddings  # Explain that the following function converts text examples into embeddings.
def embed_text(example):  # Define a function that receives one dataset example.
    embedding = model.encode(example["text"])  # Encode the example text into a dense vector embedding.
    return {"embedding": np.array(embedding, dtype=np.float32)}  # Return the embedding as a NumPy float32 array inside a dictionary.


print(f"Train 1: {embs_train[0]}")  # Print the first training example to inspect its text and label.
# Train 1: {'text': 'Sunday afternoon walking through Venice in the sun with @user ️ ️ ️ @ Abbot Kinney, Venice', 'label': 12}  # Show an example of the first training item.

embs_train = embs_train.map(embed_text, batched=False)  # Apply embed_text to each training example one by one.
embs_test = embs_test.map(embed_text, batched=False)  # Apply embed_text to each test example one by one.

# Add Faiss index which allows similarity search  # Explain that the next line builds a FAISS index for nearest-neighbor search.
embs_train.add_faiss_index("embedding")  # Add a FAISS index over the training-set embedding column.

# Run Query  # Explain that the next block selects a test example and retrieves similar training examples.
idx, knn = 1, 3  # Select the first query and 3 nearest neighbors  # Store the test example index and the number of neighbors to retrieve.

query = np.array(embs_test[idx]["embedding"], dtype=np.float32)  # Convert the selected test embedding into a NumPy float32 array.
scores, samples = embs_train.get_nearest_examples("embedding", query, k=knn)  # Retrieve the k nearest training examples using the FAISS index.

# Print Results  # Explain that the following lines display the query and retrieved neighbors.
print(f"QUERY LABEL: {embs_test[idx]['label']}")  # Print the label of the selected test query.
print(f"QUERY TEXT: {embs_test[idx]['text'][:200]} [...]\n")  # Print the first 200 characters of the query text.
print("=" * 50)  # Print a separator line.
print("Retrieved Documents:")  # Print a heading for the retrieved nearest-neighbor examples.
for score, label, text in zip(scores, samples["label"], samples["text"]):  # Loop through each retrieved score, label, and text.
    print("=" * 50)  # Print a separator line before each retrieved example.
    print(f"TEXT:\n{text[:200]} [...]")  # Print the first 200 characters of the retrieved text.
    print(f"SCORE: {score:.2f}")  # Print the similarity-search score rounded to two decimal places.
    print(f"LABEL: {label}")  # Print the label of the retrieved training example.

# QUERY LABEL: 10  # Show an example query label from the test set.
# QUERY TEXT: The calm before...... | w/ sofarsounds @user | : B. Hall.......#sofarsounds… [...]  # Show an example truncated query text.

# ==================================================  # Show an example separator line.
# Retrieved Documents:  # Show the heading for retrieved examples.
# ==================================================  # Show an example separator line.
# TEXT:  # Show the text field label for a retrieved example.
# So much love for sofarsounds &amp; sofarsoundsla! Great times this eve #sofarla #actasif @ The… [...]  # Show an example retrieved tweet text.
# SCORE: 0.85  # Show the example retrieval score for the first neighbor.
# LABEL: 4  # Show the example label for the first neighbor.
# ==================================================  # Show an example separator line.
# TEXT:  # Show the text field label for the second retrieved example.
# Couch view lookin' like: #sundaze @ The Olivian Luxury Apartment Homes [...]  # Show an example retrieved tweet text.
# SCORE: 1.11  # Show the example retrieval score for the second neighbor.
# LABEL: 1  # Show the example label for the second neighbor.
# ==================================================  # Show an example separator line.
# TEXT:  # Show the text field label for the third retrieved example.
# I ️ Faure Requiem @user oboyddd @ Walt Disney Concert Hall [...]  # Show an example retrieved tweet text.
# SCORE: 1.13  # Show the example retrieval score for the third neighbor.
# LABEL: 0  # Show the example label for the third neighbor.