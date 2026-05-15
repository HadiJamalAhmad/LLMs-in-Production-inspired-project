import slack_sdk  # Import the Slack SDK so Python can connect to Slack's Web API.
import pandas  # Import pandas so the collected Slack messages can be stored in a DataFrame.

import os  # Import os so Python can read environment variables.



token_slack = os.getenv("SLACK_TOKEN")  # Read the Slack token, returning None instead of crashing if missing.

if not token_slack:  # Check whether the token was not found.
    raise RuntimeError("SLACK_TOKEN is not set. Set it in PowerShell, then restart the VS Code kernel.")  # Show a clearer error.
client = slack_sdk.WebClient(token=token_slack)  # Create a Slack WebClient using the token for authentication.

auth = client.auth_test()  # Test the Slack token and retrieve authentication information about the current user/app.
self_user = auth["user_id"]  # Extract the authenticated user's Slack user ID from the auth response.

dm_channels_response = client.conversations_list(types="im")  # Request the list of direct-message conversations visible to the token.

all_messages = {}  # Create an empty dictionary to store messages grouped by Slack channel ID.

for channel in dm_channels_response["channels"]:  # Loop through each direct-message channel returned by Slack.
    history_response = client.conversations_history(channel=channel["id"])  # Fetch the message history for the current direct-message channel.
    all_messages[channel["id"]] = history_response["messages"]  # Store the channel's messages in the dictionary using the channel ID as the key.

txts = []  # Create an empty list that will hold rows of timestamp, user, and text values.

for channel_id, messages in all_messages.items():  # Loop through each channel ID and its list of messages.
    for message in messages:  # Loop through each individual Slack message in the current channel.
        try:  # Start a try block in case a message is missing expected fields.
            text = message["text"]  # Extract the message text.
            user = message["user"]  # Extract the Slack user ID of the message sender.
            timestamp = message["ts"]  # Extract the Slack message timestamp.
            txts.append([timestamp, user, text])  # Add the extracted timestamp, user, and text as one row.
        except Exception:  # Catch any error caused by missing fields or unexpected message structure.
            pass  # Ignore that message and continue processing the remaining messages.

slack_dataset = pandas.DataFrame(txts)  # Convert the collected rows into a pandas DataFrame.
slack_dataset.columns = ["timestamp", "user", "text"]  # Name the DataFrame columns as timestamp, user, and text.
df = slack_dataset[slack_dataset.user == self_user]  # Keep only the messages sent by the authenticated Slack user.

df[["text"]].to_parquet("slack_dataset.gzip", compression="gzip")  # Save only the text column as a gzip-compressed Parquet file.