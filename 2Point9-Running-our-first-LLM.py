from transformers import AutoModelForCausalLM, AutoTokenizer  # Import the model and tokenizer auto-classes from Hugging Face Transformers.

MODEL_NAME = "bigscience/bloom-560m"  # Store the Hugging Face model ID; change this to "bigscience/bloom-3b" if you want to test a larger BLOOM model.

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  # Load the tokenizer that matches the selected pretrained model.
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)  # Load the causal language model for text generation.

prompt = "Hello world! This is my first time running an LLM!"  # Define the input text that will be given to the language model.

input_tokens = tokenizer.encode(prompt, return_tensors="pt", padding=True)  # Convert the prompt into PyTorch token IDs that the model can process.
generated_tokens = model.generate(input_tokens, max_new_tokens=20)  # Ask the model to generate up to 20 new tokens after the input prompt.
generated_text = tokenizer.batch_decode(  # Convert the generated token IDs back into human-readable text.
    generated_tokens, skip_special_tokens=True  # Decode the tokens while removing special tokens such as padding or end-of-sequence markers.
)  # Close the batch_decode function call.

print(generated_text)  # Print the generated text output.