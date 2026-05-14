import evaluate  # Import the Hugging Face Evaluate library.

# Download a metric from Huggingface's hub  # Explain that the next line loads a metric from the Hugging Face Hub.
squad_metric = evaluate.load("squad")  # Load the SQuAD evaluation metric.

# Example from the SQuAD dataset  # Explain that the following data mimics SQuAD-style predictions and references.
predictions = [  # Start the list of model predictions.
    {  # Start the first prediction dictionary.
        "prediction_text": "Saint Bernadette",  # Store the model's predicted answer text for the first example.
        "id": "5733be284776f41900661182",  # Store the unique ID matching this prediction to its reference.
    },  # End the first prediction dictionary.
    {"prediction_text": "Salma Hayek", "id": "56d4fa2e2ccc5a1400d833cd"},  # Store the second prediction with its matching ID.
    {"prediction_text": "1000 MB", "id": "57062c2552bb89140068992c"},  # Store the third prediction with its matching ID.
]  # End the list of predictions.
references = [  # Start the list of ground-truth reference answers.
    {  # Start the first reference dictionary.
        "answers": {  # Start the answer container for the first reference.
            "text": ["Saint Bernadette Soubirous"],  # Store the accepted correct answer text for the first example.
            "answer_start": [515],  # Store the character start position of the answer in the original context.
        },  # End the answer container for the first reference.
        "id": "5733be284776f41900661182",  # Store the unique ID matching this reference to the first prediction.
    },  # End the first reference dictionary.
    {  # Start the second reference dictionary.
        "answers": {  # Start the answer container for the second reference.
            "text": ["Salma Hayek and Frida Giannini"],  # Store the accepted correct answer text for the second example.
            "answer_start": [533],  # Store the character start position of the answer in the original context.
        },  # End the answer container for the second reference.
        "id": "56d4fa2e2ccc5a1400d833cd",  # Store the unique ID matching this reference to the second prediction.
    },  # End the second reference dictionary.
    {  # Start the third reference dictionary.
        "answers": {"text": ["1000 MB"], "answer_start": [437]},  # Store the accepted answer text and start position for the third example.
        "id": "57062c2552bb89140068992c",  # Store the unique ID matching this reference to the third prediction.
    },  # End the third reference dictionary.
]  # End the list of references.

results = squad_metric.compute(  # Compute the SQuAD scores using the loaded metric.
    predictions=predictions, references=references  # Pass the predictions and references into the metric.
)  # End the metric computation call.
print(results)  # Print the computed evaluation results.
# {'exact_match': 33.333333333333336, 'f1': 79.04761904761905}  # Show the expected output containing exact match and F1 scores.