from deepeval.benchmarks import MMLU  # Imports the MMLU benchmark class from DeepEval.

from deepeval.benchmarks.tasks import MMLUTask  # Imports the MMLU task enum so you can select specific MMLU subjects.

from deepeval.models.base_model import DeepEvalBaseLLM  # Imports DeepEval's base class for wrapping a custom LLM.

import torch  # Imports PyTorch, used here to detect and manage CPU/GPU device placement.

from transformers import AutoModelForCausalLM, AutoTokenizer  # Imports Hugging Face classes for loading a causal language model and its tokenizer.


# Set up the model  # Section comment: the following class adapts a Hugging Face model to DeepEval's expected interface.

class DeepEvalLLM(DeepEvalBaseLLM):  # Defines a custom DeepEval-compatible LLM wrapper class.
    
    def __init__(self, model, tokenizer, name):  # Defines the constructor; it receives a model, tokenizer, and display name.
        
        self.model = model  # Stores the Hugging Face model instance inside the wrapper.
        
        self.tokenizer = tokenizer  # Stores the Hugging Face tokenizer instance inside the wrapper.
        
        self.name = name  # Stores a human-readable model name for DeepEval reports.

        device = torch.device(  # Creates a PyTorch device object for either GPU or CPU.
            
            "cuda" if torch.cuda.is_available() else "cpu"  # Uses CUDA GPU if available; otherwise falls back to CPU.
        
        )  # Closes the torch.device(...) call.

        self.model.to(device)  # Moves the model weights to the selected device.
        
        self.device = device  # Saves the selected device so inputs can be moved there later.

    def load_model(self):  # Implements DeepEvalBaseLLM.load_model(), which should return the wrapped model.
        
        return self.model  # Returns the Hugging Face model instance.

    def generate(self, prompt: str) -> str:  # Implements synchronous text generation from a prompt string.
        
        model = self.load_model()  # Retrieves the stored model using the wrapper method.
        
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(  # Tokenizes the prompt and converts it to PyTorch tensors.
            
            self.device  # Moves the tokenized input tensors to the same device as the model.
        
        )  # Closes the tokenizer(...).to(...) call.

        generated_ids = model.generate(  # Calls Hugging Face generation to produce output token IDs.
            
            **model_inputs, max_new_tokens=100, do_sample=True  # Passes tokenized inputs, limits new output to 100 tokens, and enables sampling.
        
        )  # Closes the model.generate(...) call.
        
        return self.tokenizer.batch_decode(generated_ids)[0]  # Decodes the generated token IDs back into text and returns the first result.

    async def a_generate(self, prompt: str) -> str:  # Implements DeepEval's asynchronous generation method.
        
        return self.generate(prompt)  # Reuses the synchronous generate method; this is simple but still blocking.

    def get_model_name(self):  # Implements DeepEvalBaseLLM.get_model_name().
        
        return self.name  # Returns the stored model name.


model = AutoModelForCausalLM.from_pretrained("gpt2")  # Loads the GPT-2 causal language model weights from Hugging Face.

tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Loads GPT-2's tokenizer from Hugging Face.

gpt2 = DeepEvalLLM(model=model, tokenizer=tokenizer, name="GPT-2")  # Wraps GPT-2 in the custom DeepEval-compatible class.

# Define benchmark with specific tasks and shots  # Section comment: configure which MMLU subjects and few-shot setting to use.

benchmark = MMLU(  # Creates an MMLU benchmark instance.
    
    tasks=[MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE, MMLUTask.ASTRONOMY],  # Evaluates only these two MMLU subject areas.
    
    n_shots=3,  # Uses 3 few-shot examples in the prompt before each benchmark question.

)  # Closes the MMLU(...) benchmark configuration.

# Run benchmark  # Section comment: execute the benchmark and print the final result.

benchmark.evaluate(model=gpt2)  # Runs the MMLU benchmark using the wrapped GPT-2 model.

print(benchmark.overall_score)  # Prints the final average MMLU accuracy across the selected tasks.

# MMLU Task Accuracy (task=high_school_computer_science): 0.0  # Example output: GPT-2 got 0% on this task.

# MMLU Task Accuracy (task=astronomy): 0.0  # Example output: GPT-2 got 0% on this task.

# Overall MMLU Accuracy: 0.0  # Example output: average accuracy across the selected MMLU tasks is 0%.