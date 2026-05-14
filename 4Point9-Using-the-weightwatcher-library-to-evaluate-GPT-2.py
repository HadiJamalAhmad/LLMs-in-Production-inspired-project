import weightwatcher as ww  # Import the WeightWatcher library and give it the shorter alias ww.
from transformers import GPT2Model  # Import the GPT2Model class from Hugging Face Transformers.

gpt2_model = GPT2Model.from_pretrained("gpt2")  # Load the pretrained GPT-2 base model using the "gpt2" checkpoint.
gpt2_model.eval()  # Put the GPT-2 model into evaluation mode instead of training mode.

watcher = ww.WeightWatcher(model=gpt2_model)  # Create a WeightWatcher object that will analyze the GPT-2 model's weights.
details = watcher.analyze(plot=False)  # Analyze the model weights without generating plots.
print(details.head())  # Print the first five rows of the analysis results table.
#    layer_id       name         D  ...      warning        xmax        xmin  # Show the column names and sample output from the WeightWatcher results.
# 0         2  Embedding  0.076190  ... over-trained 3837.188332    0.003564  # Show the first analyzed layer, an embedding layer, with its metrics.
# 1         8     Conv1D  0.060738  ...              2002.124419  108.881419  # Show the second displayed analyzed layer, a Conv1D layer, with its metrics.
# 2         9     Conv1D  0.037382  ...               712.127195   46.092445  # Show the third displayed analyzed layer, another Conv1D layer, with its metrics.
# 3        14     Conv1D  0.042383  ...              1772.850274   95.358278  # Show the fourth displayed analyzed layer and its metrics.
# 4        15     Conv1D  0.062197  ...               626.655218   23.727908  # Show the fifth displayed analyzed layer and its metrics.