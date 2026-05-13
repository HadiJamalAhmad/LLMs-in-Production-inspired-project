#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os  # Imports Python's operating-system utilities, used here for file paths and file-existence checks.
import nltk  # Imports the NLTK library, used here for tokenization resources, corpora, and language modeling.

# 1) Set the REAL path to the folder containing hamlet.txt
corpus_root = r"C:\Users\HADI\Downloads"   # change if needed  # Stores the folder path where hamlet.txt is expected to be located.
file_id = "hamlet.txt"  # Stores the exact filename that will be read from the corpus folder.

# 2) Validate file exists
full_path = os.path.join(corpus_root, file_id)  # Combines the folder path and filename into one complete file path.
if not os.path.exists(full_path):  # Checks whether the complete path actually exists on your computer.
    raise FileNotFoundError(  # Raises an error and stops the program if hamlet.txt is not found.
        f"File not found!\nExpected: {full_path}\n\n"  # Creates the first part of the error message showing the expected file path.
        f"Fix: Put hamlet.txt there or update corpus_root."  # Creates the second part of the error message explaining how to fix the problem.
    )  # Closes the FileNotFoundError message construction.

# 3) Download NLTK tokenizer resources
download_dir = r"C:\Users\HADI\nltk_data"  # Sets the folder where NLTK tokenizer data will be downloaded and stored.
nltk.download("punkt_tab", download_dir=download_dir)  # Downloads the punkt_tab tokenizer resource into the chosen NLTK data folder.
nltk.download("punkt", download_dir=download_dir)  # Downloads the punkt tokenizer resource into the chosen NLTK data folder.

if download_dir not in nltk.data.path:  # Checks whether the custom NLTK data folder is already known to NLTK.
    nltk.data.path.append(download_dir)  # Adds the custom NLTK data folder so NLTK can find the downloaded tokenizer resources.

from nltk.corpus.reader import PlaintextCorpusReader  # Imports the reader used to treat local .txt files as an NLTK corpus.
from nltk.util import everygrams  # Imports the utility that generates n-grams of multiple lengths from a token sequence.
from nltk.lm.preprocessing import pad_both_ends, flatten, padded_everygram_pipeline  # Imports preprocessing helpers for padding, flattening, and preparing n-gram training data.
from nltk.lm import MLE  # Imports the Maximum Likelihood Estimation n-gram language model class.



# 4) Create corpus from .txt files in the chosen folder
my_corpus = PlaintextCorpusReader(corpus_root, r".*\.txt")  # Creates a corpus reader for all .txt files inside corpus_root.

# 5) Read and materialize sentences
sentences = list(my_corpus.sents(fileids=file_id))  # Reads tokenized sentences from hamlet.txt and converts them into a list.

# Count sentences
print("Number of sentences:", len(sentences))  # Prints the total number of tokenized sentences found in hamlet.txt.


# Print sentences
for sent in sentences:  # Loops through every tokenized sentence in the sentence list.
    print(sent)  # Prints the current sentence as a list of word tokens.

# Pad a specific sentence with start/end markers
padded_bigrams = list(pad_both_ends(sentences[1104], n=2))  # Adds start and end padding markers to sentence 1104 for bigram-style context.
print(list(everygrams(padded_bigrams, max_len=3)))  # Prints all 1-grams, 2-grams, and 3-grams generated from the padded sentence.

# Flatten padded sentences
flattened_tokens = list(  # Starts creating a list of all padded tokens from all sentences.
    flatten(  # Flattens multiple padded sentence iterables into one continuous token stream.
        pad_both_ends(sent, n=2)  # Pads the current sentence with start and end markers for bigram-style context.
        for sent in sentences  # Repeats the padding operation for every sentence in the corpus.
    )  # Closes the flatten function call.
)  # Converts the flattened token stream into a list.
print(flattened_tokens)  # Prints the complete flattened list of padded tokens.

# Create training data and vocab for a trigram model
train, vocab = padded_everygram_pipeline(3, sentences)  # Creates padded everygram training data and vocabulary data for a trigram model.

# Train Maximum Likelihood Estimation model
lm = MLE(3)  # Creates a trigram Maximum Likelihood Estimation language model.

print(len(lm.vocab))  # likely 0 before fitting  # Prints the vocabulary size before training the model.

lm.fit(train, vocab)  # Fits the trigram language model using the training n-grams and vocabulary.

print(lm.vocab)  # Prints the trained vocabulary object.
print(len(lm.vocab))  # Prints the size of the trained vocabulary.

# Generate text conditioned on previous words
print(lm.generate(6, ["to", "be"]))  # Generates 6 tokens using "to be" as the starting context.

# Vocabulary lookup examples
print(list(lm.vocab.lookup(sentences[1104])))  # Looks up the tokens from sentence 1104 in the model vocabulary.
print(list(lm.vocab.lookup(["aliens", "from", "Mars"])))  # Looks up unseen or unusual tokens and maps unknown ones as needed.

# N-gram counts
print(lm.counts)  # Prints the model's stored n-gram counts.
print(lm.counts[["to"]]["be"])  # Prints how often "be" appears after the context "to" in the trained model.

# Probabilities
print(lm.score("be"))  # Prints the unigram probability score for the word "be".
print(lm.score("be", ["to"]))  # Prints the probability of "be" given the previous word "to".
print(lm.score("be", ["not", "to"]))  # Prints the probability of "be" given the previous two words "not to".

# Log probabilities
print(lm.logscore("be"))  # Prints the log probability score for the word "be".
print(lm.logscore("be", ["to"]))  # Prints the log probability of "be" given the previous word "to".
print(lm.logscore("be", ["not", "to"]))  # Prints the log probability of "be" given the previous two words "not to".

# Entropy and perplexity
test = [("to", "be"), ("or", "not"), ("to", "be")]  # Creates a small test sequence of bigrams for evaluation.
print("lm.entropy(test):",lm.entropy(test))  # Prints the entropy of the model on the test n-grams.
print("lm.perplexity(test):",lm.perplexity(test))  # Prints the perplexity of the model on the test n-grams.