# Import the ROUGE scorer module from the rouge_score package.
from rouge_score import rouge_scorer

# Define the reference text that the prediction will be compared against.
target = "The game 'The Legend of Zelda' follows the adventures of the \
    hero Link in the magical world of Hyrule."  # Continue and complete the target reference text.
# Define the generated/predicted text that will be evaluated.
prediction = "Link embarks on epic quests and battles evil forces to \
    save Princess Zelda and restore peace in the land of Hyrule."  # Continue and complete the prediction text.

# Example N-gram where N=1 and also using the longest common subsequence
# Create a ROUGE scorer for ROUGE-1 and ROUGE-L, using stemming for word-form normalization.
scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
# Compare the target reference text with the prediction and store the ROUGE scores.
scores = scorer.score(target, prediction)
# Print the resulting ROUGE precision, recall, and F-measure scores.
print(scores)
# {'rouge1': Score(precision=0.28571428, recall=0.31578947, fmeasure=0.3),
# 'rougeL': Score(precision=0.238095238, recall=0.26315789, fmeasure=0.25)}