import os  # Import the os module to check whether directories exist and create them if needed.
from pathlib import Path  # Import Path to work with filesystem paths in a clean, object-oriented way.

import transformers  # Import Hugging Face Transformers to use PreTrainedTokenizerFast later.
from tokenizers import ByteLevelBPETokenizer, SentencePieceBPETokenizer  # Import two tokenizer classes for Byte-Level BPE and SentencePiece-style BPE.
from tokenizers.processors import BertProcessing  # Import BertProcessing to add BERT/RoBERTa-style special-token processing.

# Initialize the txts to train from  # Explain that the next line collects text files used as tokenizer training data.
paths = [str(x) for x in Path("./data/").glob("**/*.txt")]  # Find every .txt file under ./data/ recursively and convert each path to a string.

# Train a Byte-Pair Encoding tokenizer  # Explain that the following section trains a BPE tokenizer.
bpe_tokenizer = ByteLevelBPETokenizer()  # Create an empty byte-level BPE tokenizer.

bpe_tokenizer.train(  # Start training the byte-level BPE tokenizer.
    files=paths,  # Pass the list of text files that will be used as training data.
    vocab_size=52_000,  # Set the target vocabulary size to 52,000 tokens.
    min_frequency=2,  # Only include token pairs that appear at least twice.
    show_progress=True,  # Display training progress while the tokenizer is being trained.
    special_tokens=[  # Start the list of special tokens to reserve in the vocabulary.
        "<s>",  # Add a beginning-of-sequence token.
        "<pad>",  # Add a padding token.
        "</s>",  # Add an end-of-sequence token.
        "<unk>",  # Add an unknown-token marker.
        "<mask>",  # Add a mask token, often used for masked-language-modeling tasks.
    ],  # End the list of special tokens.
)  # Finish the tokenizer training call.

token_dir = "./models/tokenizers/bytelevelbpe/"  # Define the directory where the byte-level BPE tokenizer files will be saved.
if not os.path.exists(token_dir):  # Check whether the tokenizer directory does not already exist.
    os.makedirs(token_dir)  # Create the tokenizer directory, including any missing parent folders.
bpe_tokenizer.save_model(token_dir)  # Save the trained byte-level BPE tokenizer files to the chosen directory.

bpe_tokenizer = ByteLevelBPETokenizer(  # Reload the byte-level BPE tokenizer from saved files.
    f"{token_dir}vocab.json",  # Provide the path to the saved vocabulary JSON file.
    f"{token_dir}merges.txt",  # Provide the path to the saved BPE merges file.
)  # Finish reloading the byte-level BPE tokenizer.

example_text = "This sentence is getting encoded by a tokenizer."  # Define a sample sentence to test the tokenizer.
print(bpe_tokenizer.encode(example_text).tokens)  # Encode the sample text and print the resulting token strings.
# ['This', 'Ġsentence', 'Ġis', 'Ġgetting', 'Ġenc', \  # Show an example list of byte-level BPE tokens.
#  'oded', 'Ġby', 'Ġa', 'Ġto', 'ken', 'izer', '.']  # Continue the example token list.
print(bpe_tokenizer.encode(example_text).ids)  # Encode the sample text and print the corresponding token IDs.
# [2666, 5651, 342, 1875, 4650, 10010, 504, 265, \  # Show an example list of token IDs.
# 285, 1507, 13035, 18]  # Continue the example token ID list.

bpe_tokenizer._tokenizer.post_processor = BertProcessing(  # Add post-processing so encoded sequences receive special tokens.
    ("</s>", bpe_tokenizer.token_to_id("</s>")),  # Define the end-of-sequence token and look up its token ID.
    ("<s>", bpe_tokenizer.token_to_id("<s>")),  # Define the beginning-of-sequence token and look up its token ID.
)  # Finish configuring the BERT-style post-processor.
bpe_tokenizer.enable_truncation(max_length=512)  # Enable truncation so encoded sequences are limited to 512 tokens.


# Train a Sentencepiece Tokenizer  # Explain that the following section trains a SentencePiece-style BPE tokenizer.
special_tokens = [  # Start defining the special tokens used for the SentencePiece-style tokenizer.
    "<s>",  # Add a beginning-of-sequence token.
    "<pad>",  # Add a padding token.
    "</s>",  # Add an end-of-sequence token.
    "<unk>",  # Add an unknown-token marker.
    "<cls>",  # Add a classification token.
    "<sep>",  # Add a separator token.
    "<mask>",  # Add a mask token.
]  # End the special token list.
sentencepiece_tokenizer = SentencePieceBPETokenizer()  # Create an empty SentencePiece-style BPE tokenizer.

sentencepiece_tokenizer.train(  # Start training the SentencePiece-style BPE tokenizer.
    files=paths,  # Pass the list of text files used as training data.
    vocab_size=4000,  # Set the target vocabulary size to 4,000 tokens.
    min_frequency=2,  # Only include token pieces that appear at least twice.
    show_progress=True,  # Display progress during tokenizer training.
    special_tokens=special_tokens,  # Reserve the previously defined special tokens in the vocabulary.
)  # Finish the SentencePiece-style tokenizer training call.

token_dir = "./models/tokenizers/sentencepiece/"  # Define the directory where the SentencePiece-style tokenizer files will be saved.
if not os.path.exists(token_dir):  # Check whether this tokenizer directory does not already exist.
    os.makedirs(token_dir)  # Create the tokenizer directory if it is missing.
sentencepiece_tokenizer.save_model(token_dir)  # Save the trained SentencePiece-style tokenizer files.

# convert  # Explain that the following section wraps the tokenizer as a Transformers fast tokenizer.
tokenizer = transformers.PreTrainedTokenizerFast(  # Create a Transformers-compatible fast tokenizer from the trained tokenizer object.
    tokenizer_object=sentencepiece_tokenizer,  # Pass the trained SentencePiece-style tokenizer object.
    model_max_length=512,  # Set the maximum model input length to 512 tokens.
    special_tokens=special_tokens,  # Pass the list of special tokens to the fast tokenizer wrapper.
)  # Finish creating the fast tokenizer.
tokenizer.bos_token = "<s>"  # Set the beginning-of-sequence token string.
tokenizer.bos_token_id = sentencepiece_tokenizer.token_to_id("<s>")  # Set the beginning-of-sequence token ID.
tokenizer.pad_token = "<pad>"  # Set the padding token string.
tokenizer.pad_token_id = sentencepiece_tokenizer.token_to_id("<pad>")  # Set the padding token ID.
tokenizer.eos_token = "</s>"  # Set the end-of-sequence token string.
tokenizer.eos_token_id = sentencepiece_tokenizer.token_to_id("</s>")  # Set the end-of-sequence token ID.
tokenizer.unk_token = "<unk>"  # Set the unknown-token string.
tokenizer.unk_token_id = sentencepiece_tokenizer.token_to_id("<unk>")  # Set the unknown-token ID.
tokenizer.cls_token = "<cls>"  # Set the classification token string.
tokenizer.cls_token_id = sentencepiece_tokenizer.token_to_id("<cls>")  # Set the classification token ID.
tokenizer.sep_token = "<sep>"  # Set the separator token string.
tokenizer.sep_token_id = sentencepiece_tokenizer.token_to_id("<sep>")  # Set the separator token ID.
tokenizer.mask_token = "<mask>"  # Set the mask token string.
tokenizer.mask_token_id = sentencepiece_tokenizer.token_to_id("<mask>")  # Set the mask token ID.
# and save for later!  # Explain that the tokenizer is saved so it can be reused later.
tokenizer.save_pretrained(token_dir)  # Save the Transformers-compatible tokenizer configuration and files.

print(tokenizer.tokenize(example_text))  # Tokenize the sample text and print the resulting token strings.
# ['▁This', '▁s', 'ent', 'ence', '▁is', '▁', 'g', 'et', 'tin', 'g', '▁',  # Show an example SentencePiece-style tokenization output.
#  'en', 'co', 'd', 'ed', '▁', 'b', 'y', '▁a', '▁', 't', 'ok', 'en',  # Continue the example token output.
#  'iz', 'er', '.']  # Finish the example token output.
print(tokenizer.encode(example_text))  # Encode the sample text and print the resulting token IDs.
# [814, 1640, 609, 203, 1810, 623, 70, \  # Show an example list of encoded token IDs.
#  351, 148, 371, 125, 146, 2402, 959, 632]  # Finish the example encoded ID list.