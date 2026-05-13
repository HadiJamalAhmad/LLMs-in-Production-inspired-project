# Import the Hugging Face Datasets helper used to load datasets from the Hub.
from datasets import load_dataset
# Import the Hugging Face Transformers auto classes for tokenization and causal language modeling.
from transformers import AutoTokenizer, AutoModelForCausalLM

# SuperGlue has mutliple test datasets, options are boolq,
# cb, copa, multirc, record, rte, wic, wsc, wsc.fixed, axb, axg
# Load the MultiRC validation split from the SuperGLUE benchmark dataset.
dataset = load_dataset("super_glue", "multirc", split="validation")
# Print the first row of the validation dataset to inspect its structure.
print(dataset[0])

# {
#   "paragraph": "What causes a change in motion? The application of a force."
#     " Any time an object changes motion, a force has been applied. In what "
#     "ways can this happen? Force can cause an object at rest to start "
#     "moving. Forces can cause objects to speed up or slow down. Forces can "
#     "cause a moving object to stop. Forces can also cause a change in "
#     "direction. In short, forces cause changes in motion. The moving "
#     "object may change its speed, its direction, or both. We know that "
#     "changes in motion require a force. We know that the size of the force "
#     "determines the change in motion. How much an objects motion changes "
#     "when a force is applied depends on two things. It depends on the "
#     "strength of the force. It also depends on the objects mass. Think "
#     "about some simple tasks you may regularly do. You may pick up a "
#     "baseball. This requires only a very small force. ",
#   "question": "Would the mass of a baseball affect how much force you have "
#     "to use to pick it up?",
#   "answer": "No",
#   "idx": {"paragraph": 0, "question": 0, "answer": 0},
#   "label": 0,
# }

# Store the model checkpoint name to use for both tokenizer and model loading.
model = "bigscience/bloomz-560m"  # Update with your model of choice

# Load the tokenizer associated with the selected model checkpoint.
tokenizer = AutoTokenizer.from_pretrained(model)
# Load the causal language model associated with the selected model checkpoint.
model = AutoModelForCausalLM.from_pretrained(model)


# Iterate through every row in the dataset.
for row in dataset:
    # replace this with the correct input for your benchmark
    # Build the prompt text from the paragraph and question fields of the current row.
    input_text = (
        # Add the paragraph and question into one formatted input string.
        f'Paragraph: {row["paragraph"]}\nQuestion: {row["question"]}'
    )
    # Tokenize the input text and return PyTorch tensors, then extract the input IDs.
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    # Generate up to 20 new tokens from the model using the tokenized input.
    outputs = model.generate(input_ids, max_new_tokens=20)
    # Store the length of the input tokens so the prompt can be removed from the decoded output.
    input_length = input_ids.shape[1]  # We use this to trim out the input
    # Decode only the newly generated tokens after the original input tokens.
    results = tokenizer.decode(outputs[0][input_length:])
    # Print the reference answer from the dataset row.
    print(row["answer"])
    # Print the model-generated answer.
    print(results)

# No
#  No</s>
# Yes
#  No</s>
# Less the mass, less the force applied
#  No</s>
# It depends on the shape of the baseball
#  No</s>
# Strength
#  Force</s>
# A force
#  Force</s>
# No
#  Yes</s>