import pandas as pd  # Import pandas to manipulate and save the dataset.
from datasets import load_dataset  # Import Hugging Face Datasets to download the public Slack-style dataset.

dataset = load_dataset("spencer/python_slack", split="train")  # Load the public Python Slack dataset from Hugging Face.

df = dataset.to_pandas()  # Convert the Hugging Face dataset into a pandas DataFrame.

df = df.rename(columns={"sentences": "text"})  # Rename the main message column to "text" so it matches your book's expected format.

df = df[["text"]]  # Keep only the text column, matching the original slack_dataset.gzip structure.

df = df.dropna()  # Remove rows where the text is missing.

df["text"] = df["text"].astype(str).str.strip()  # Convert text to string and remove leading/trailing whitespace.

df = df[df["text"].str.len() > 0]  # Remove empty messages.

df.to_parquet("slack_dataset.gzip", compression="gzip")  # Save the replacement dataset using the same filename as the book code.

print(df.shape)  # Print the number of rows and columns saved.

print(df.head())  # Print the first few rows to verify the dataset.