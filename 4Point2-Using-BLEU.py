# Import NLTK's BLEU score module and give it the shorter alias bleu.
import nltk.translate.bleu_score as bleu

# Create a list of reference sentences, where each reference is tokenized into words.
target = [
    # First reference sentence, split into a list of tokens.
    "The game 'The Legend of Zelda' follows the adventures of the \
    hero Link in the magical world of Hyrule.".split(),
    # Second reference sentence, split into a list of tokens.
    "Link goes on awesome quests and battles evil forces to \
    save Princess Zelda and restore peace to Hyrule.".split(),
]
# Create the predicted sentence, split into a list of tokens.
prediction = "Link embarks on epic quests and battles evil forces to \
    save Princess Zelda and restore peace in the land of Hyrule.".split()


# Calculate the sentence-level BLEU score between the references and the prediction.
score = bleu.sentence_bleu(target, prediction)
# Print the BLEU score.
print(score)  # 0.6187934993051339