import torch  # Import PyTorch so we can check whether CUDA/GPU is available and create a device object.
from transformers import pipeline  # Import Hugging Face's pipeline helper for easy text-generation inference.
from datasets import Dataset, load_dataset  # Import Dataset for rebuilding datasets and load_dataset for downloading a dataset from Hugging Face Hub.
from evaluate import evaluator  # Import Hugging Face Evaluate's evaluator factory for task-level model evaluation.
import evaluate  # Import the main Evaluate library so we can load metrics/measurements such as "regard".
import pandas as pd  # Import pandas so we can convert nested metric outputs into DataFrames for averaging.

device = 0 if torch.cuda.is_available() else -1 

# Pull model, data, and metrics  # This section loads the generation model, the bias-prompt dataset, and the regard metric.
pipe = pipeline(  # Create a Hugging Face text-generation pipeline.
    "text-generation", model="gpt2", pad_token_id=50256, device=device  # Use GPT-2, set GPT-2's end-of-text token as the padding token, and place the pipeline on the selected device.
)  # Finish constructing the text-generation pipeline.
wino_bias = load_dataset("sasha/wino_bias_prompt1", split="test")  # Load the test split of the WinoBias prompt dataset from Hugging Face Hub.
polarity = evaluate.load("regard")  # Load the Regard measurement, which scores the polarity/regard of generated text.
task_evaluator = evaluator("text-generation")  # Create a text-generation evaluator that can run a generation pipeline over a dataset and score outputs.

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
print("GPT-2 pipeline device:", pipe.device)

# Prepare dataset  # This section defines a helper function to filter and format prompts by pronoun.
def prepare_dataset(wino_bias, pronoun):  # Define a function that takes the full WinoBias dataset and a target pronoun such as "she" or "he".
    data = wino_bias.filter(  # Filter the dataset so only examples with the requested bias pronoun remain.
        lambda example: example["bias_pronoun"] == pronoun  # Keep only rows whose "bias_pronoun" field equals the pronoun argument.
    ).shuffle()  # Randomly shuffle the filtered dataset so the order is not fixed.
    df = data.to_pandas()  # Convert the Hugging Face Dataset into a pandas DataFrame for easier column manipulation.
    df["prompts"] = df["prompt_phrase"] + " " + df["bias_pronoun"]  # Build the final prompt text by joining the prompt phrase and the pronoun.
    return Dataset.from_pandas(df)  # Convert the pandas DataFrame back into a Hugging Face Dataset and return it.


female_prompts = prepare_dataset(wino_bias, "she")  # Create the dataset subset containing prompts ending with the female pronoun "she".
male_prompts = prepare_dataset(wino_bias, "he")  # Create the dataset subset containing prompts ending with the male pronoun "he".

# Run through evaluation pipeline  # This section generates continuations and evaluates them with the Regard metric.
female_results = task_evaluator.compute(  # Run the text-generation evaluator on the female-pronoun prompts.
    model_or_pipeline=pipe,  # Use the GPT-2 text-generation pipeline created above.
    data=female_prompts,  # Provide the female-pronoun prompt dataset as input data.
    input_column="prompts",  # Tell the evaluator that the prompt text is stored in the "prompts" column.
    metric=polarity,  # Score the generated outputs using the Regard measurement.
)  # Store the evaluator output for the female-pronoun prompts.
male_results = task_evaluator.compute(  # Run the text-generation evaluator on the male-pronoun prompts.
    model_or_pipeline=pipe,  # Use the same GPT-2 text-generation pipeline.
    data=male_prompts,  # Provide the male-pronoun prompt dataset as input data.
    input_column="prompts",  # Tell the evaluator that the prompt text is stored in the "prompts" column.
    metric=polarity,  # Score the generated outputs using the same Regard measurement.
)  # Store the evaluator output for the male-pronoun prompts.


# Analyze results  # This section converts the nested Regard output into a tabular format.
def flatten_results(results):  # Define a helper function that converts evaluator results into a pandas DataFrame.
    flattened_results = []  # Create an empty list that will store one dictionary of scores per generated completion.
    for result in results["regard"]:  # Loop over the Regard scores for each generated text output.
        item_dict = {}  # Create an empty dictionary for the current generated output's label-to-score mapping.
        for item in result:  # Loop over each label-score item, such as positive, negative, neutral, or other.
            item_dict[item["label"]] = item["score"]  # Store the score under its corresponding Regard label.
        flattened_results.append(item_dict)  # Add the current output's score dictionary to the list.

    return pd.DataFrame(flattened_results)  # Convert the list of dictionaries into a pandas DataFrame and return it.


# Print the mean polarity scores  # This section prints average Regard scores across all generated completions for each group.
print(flatten_results(female_results).mean())  # Flatten female-prompt Regard results and print the mean score for each Regard label.
# positive    0.129005  # Example expected mean positive Regard score for female-pronoun prompts.
# negative    0.391423  # Example expected mean negative Regard score for female-pronoun prompts.
# neutral     0.331425  # Example expected mean neutral Regard score for female-pronoun prompts.
# other       0.148147  # Example expected mean other Regard score for female-pronoun prompts.
print(flatten_results(male_results).mean())  # Flatten male-prompt Regard results and print the mean score for each Regard label.
# positive    0.118647  # Example expected mean positive Regard score for male-pronoun prompts.
# negative    0.406649  # Example expected mean negative Regard score for male-pronoun prompts.
# neutral     0.322766  # Example expected mean neutral Regard score for male-pronoun prompts.
# other       0.151938  # Example expected mean other Regard score for male-pronoun prompts.