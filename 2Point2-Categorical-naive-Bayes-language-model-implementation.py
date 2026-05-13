#!/usr/bin/env python  # Tells the operating system to run this file using the default Python interpreter.
# coding: utf-8  # Declares that this Python source file uses UTF-8 character encoding.

# In[1]:  # Marks this as the first notebook-style code cell when exported from Jupyter.


import re  # Imports Python's regular expression module for text pattern matching and substitution.

def process_utt(utt):  # Defines a function that processes one utterance or sentence.
    utt = utt.lower()  # Converts the entire utterance to lowercase so words are treated consistently.
    utt = re.sub(r'[^\w\s]', '', utt)  # Removes punctuation by replacing non-word and non-space characters with nothing.
    return utt.split()  # Splits the cleaned utterance into a list of words.

def lookup(freqs, word, label):  # Defines a helper function to look up a word-label frequency.
    return freqs.get((word, label), 0)  # Returns the frequency of the word-label pair, or 0 if it is missing.



from nltk.corpus.reader import PlaintextCorpusReader  # Imports NLTK's reader for plaintext corpus files.
import numpy as np  # Imports NumPy and gives it the shorter alias np for numerical operations.


corpus_root = r"C:\Users\HADI\Downloads"  # Stores the folder path where the text corpus file is located.
my_corpus = PlaintextCorpusReader(corpus_root, r".*\.txt")  # Creates a corpus reader for all .txt files in the folder.
sents = my_corpus.sents(fileids="hamlet.txt")  # Reads the sentences from the file named hamlet.txt.


def count_utts(result, utts, ys):  # Defines a function to count word frequencies by label.
    """  # Starts the function documentation string.
    Input:  # Introduces the input section of the documentation.
        result: a dictionary that is used to map each pair to its frequency  # Describes the result dictionary input.
        utts: a list of utts  # Describes the list of utterances input.
        ys: a list of the sentiment of each utt (either 0 or 1)  # Describes the label list input.
    Output:  # Introduces the output section of the documentation.
        result: a dictionary mapping each pair to its frequency  # Describes the returned dictionary.
    """  # Ends the function documentation string.

    for y, utt in zip(ys, utts):  # Loops through each label and its corresponding utterance together.
        for word in process_utt(utt):  # Processes the utterance and loops through each resulting word.
            # define the key, which is the word and label tuple  # Explains that the dictionary key combines the word and label.
            pair = (word, y)  # Creates a tuple containing the word and its label.

            # if the key exists in the dictionary, increment the count  # Explains the next conditional case.
            if pair in result:  # Checks whether this word-label pair is already in the dictionary.
                result[pair] += 1  # Increases the stored count for this word-label pair by 1.

            # if the key is new, add it to the dict and set the count to 1  # Explains the alternative case.
            else:  # Runs when the word-label pair is not already in the dictionary.
                result[pair] = 1  # Adds the new word-label pair with an initial count of 1.

    return result  # Returns the completed frequency dictionary.


result = {}  # Creates an empty dictionary to store word-label counts.
utts = [" ".join(sent) for sent in sents]  # Converts each tokenized sentence into a single string.
ys = [sent.count("be") > 0 for sent in sents]  # Labels each sentence as True if it contains the word "be".
count_utts(result, utts, ys)  # Counts word-label frequencies and stores them in result.

freqs = count_utts({}, utts, ys)  # Creates a fresh frequency dictionary from the utterances and labels.
lookup(freqs, "be", True)  # Looks up how often the word "be" appears with the True label.
for k, v in freqs.items():  # Loops through every key-value pair in the frequency dictionary.
    if "be" in k:  # Checks whether the string "be" appears inside the key tuple.
        print(f"{k}:{v}")  # Prints the matching key and its frequency value.


def train_naive_bayes(freqs, train_x, train_y):  # Defines the function that trains a Naive Bayes classifier.
    """  # Starts the function documentation string.
    Input:  # Introduces the input section of the documentation.
        freqs: dictionary from (word, label) to how often the word appears  # Describes the frequency dictionary input.
        train_x: a list of utts  # Describes the training utterances input.
        train_y: a list of labels correponding to the utts (0,1)  # Describes the labels corresponding to the utterances.
    Output:  # Introduces the output section of the documentation.
        logprior: the log prior.  # Describes the log prior output.
        loglikelihood: the log likelihood of you Naive bayes equation.  # Describes the log-likelihood dictionary output.
    """  # Ends the function documentation string.
    loglikelihood = {}  # Creates an empty dictionary to store each word's log likelihood.
    logprior = 0  # Initializes the log prior value to zero.

    # calculate V, the number of unique words in the vocabulary  # Explains that the next lines compute vocabulary size.
    vocab = set([pair[0] for pair in freqs.keys()])  # Builds a set of unique words from the frequency dictionary keys.
    V = len(vocab)  # Stores the number of unique vocabulary words.

    # calculate N_pos and N_neg  # Explains that the next section counts positive and negative word totals.
    N_pos = N_neg = 0  # Initializes both positive and negative word counts to zero.
    for pair in freqs.keys():  # Loops through every word-label pair in the frequency dictionary.
        # if the label is positive (greater than zero)  # Explains the condition for positive labels.
        if pair[1] > 0:  # Checks whether the label is positive or True.
            # Increment the number of positive words (word, label)  # Explains the positive count update.
            N_pos += lookup(freqs, pair[0], True)  # Adds this word's positive frequency to the total positive count.

        # else, the label is negative  # Explains the alternative negative-label case.
        else:  # Runs when the label is not positive.
            # increment the number of negative words (word,label)  # Explains the negative count update.
            N_neg += lookup(freqs, pair[0], False)  # Adds this word's negative frequency to the total negative count.

    # Calculate D, the number of documents  # Explains that the next line counts total training examples.
    D = len(train_y)  # Stores the number of training labels, equal to the number of training utterances.

    # Calculate the number of positive documents  # Explains that the next line counts positive examples.
    D_pos = sum(train_y)  # Counts how many training labels are True or 1.

    # Calculate the number of negative documents  # Explains that the next line counts negative examples.
    D_neg = D - D_pos  # Computes the number of negative examples by subtraction.

    # Calculate logprior  # Explains that the next line calculates the class prior ratio in log form.
    logprior = np.log(D_pos) - np.log(D_neg)  # Computes log(P_positive) minus log(P_negative).

    # For each word in the vocabulary...  # Explains that the next loop computes likelihoods for every word.
    for word in vocab:  # Loops through every unique word in the vocabulary.
        # get the positive and negative frequency of the word  # Explains that the next lines fetch word counts by class.
        freq_pos = lookup(freqs, word, 1)  # Gets the frequency of the word in positive examples.
        freq_neg = lookup(freqs, word, 0)  # Gets the frequency of the word in negative examples.

        # calculate the probability that each word is positive, and negative  # Explains the Laplace-smoothed probability calculations.
        p_w_pos = (freq_pos + 1) / (N_pos + V)  # Computes the smoothed probability of the word given the positive class.
        p_w_neg = (freq_neg + 1) / (N_neg + V)  # Computes the smoothed probability of the word given the negative class.

        # calculate the log likelihood of the word  # Explains that the next line stores the word's class evidence score.
        loglikelihood[word] = np.log(p_w_pos / p_w_neg)  # Stores the log ratio of positive probability to negative probability.

    return logprior, loglikelihood  # Returns the trained prior and word log-likelihood dictionary.


def naive_bayes_predict(utt, logprior, loglikelihood):  # Defines a function that predicts a score for one utterance.
    """  # Starts the function documentation string.
    Input:  # Introduces the input section of the documentation.
        utt: a string  # Describes the utterance input.
        logprior: a number  # Describes the log prior input.
        loglikelihood: a dictionary of words mapping to numbers  # Describes the word-score dictionary input.
    Output:  # Introduces the output section of the documentation.
        p: the sum of all the logliklihoods + logprior  # Describes the returned prediction score.
    """  # Ends the function documentation string.
    # process the utt to get a list of words  # Explains that the utterance must be cleaned and tokenized.
    word_l = process_utt(utt)  # Converts the utterance into a list of lowercase words without punctuation.

    # initialize probability to zero  # Explains the starting value for the prediction score.
    p = 0  # Initializes the prediction score.

    # add the logprior  # Explains that the class prior is included before word evidence.
    p += logprior  # Adds the trained log prior to the score.

    for word in word_l:  # Loops through each processed word in the utterance.
        # check if the word exists in the loglikelihood dictionary  # Explains the vocabulary check.
        if word in loglikelihood:  # Checks whether the word was seen during training.
            # add the log likelihood of that word to the probability  # Explains the score update for known words.
            p += loglikelihood[word]  # Adds the word's log-likelihood score to the prediction score.

    return p  # Returns the final Naive Bayes score.


def test_naive_bayes(test_x, test_y, logprior, loglikelihood):  # Defines a function to evaluate Naive Bayes accuracy.
    """  # Starts the function documentation string.
    Input:  # Introduces the input section of the documentation.
        test_x: A list of utts  # Describes the test utterances input.
        test_y: the corresponding labels for the list of utts  # Describes the true labels input.
        logprior: the logprior  # Describes the trained log prior input.
        loglikelihood: a dictionary with the loglikelihoods for each word  # Describes the trained word-score dictionary input.
    Output:  # Introduces the output section of the documentation.
        accuracy: (# of utts classified correctly)/(total # of utts)  # Describes the returned accuracy.
    """  # Ends the function documentation string.
    accuracy = 0  # return this properly  # Initializes accuracy before calculating it later.

    y_hats = []  # Creates an empty list to store predicted labels.
    for utt in test_x:  # Loops through each test utterance.
        # if the prediction is > 0  # Explains the decision threshold.
        if naive_bayes_predict(utt, logprior, loglikelihood) > 0:  # Predicts positive if the score is greater than zero.
            # the predicted class is 1  # Explains the positive prediction assignment.
            y_hat_i = 1  # Stores the predicted label as 1.
        else:  # Runs when the prediction score is not greater than zero.
            # otherwise the predicted class is 0  # Explains the negative prediction assignment.
            y_hat_i = 0  # Stores the predicted label as 0.

        # append the predicted class to the list y_hats  # Explains that the prediction is saved.
        y_hats.append(y_hat_i)  # Adds the predicted label to the predictions list.

    # error = avg of the abs vals of the diffs between y_hats and test_y  # Explains how the classification error is calculated.
    error = sum(  # Starts calculating the average absolute difference between predictions and true labels.
        [abs(y_hat - test) for y_hat, test in zip(y_hats, test_y)]  # Builds a list of prediction mistakes as absolute differences.
    ) / len(y_hats)  # Divides the total error by the number of predictions.

    # Accuracy is 1 minus the error  # Explains how accuracy is derived from error.
    accuracy = 1 - error  # Converts the average error into accuracy.

    return accuracy  # Returns the final accuracy score.


if __name__ == "__main__":  # Ensures the following code runs only when this file is executed directly.
    logprior, loglikelihood = train_naive_bayes(freqs, utts, ys)  # Trains the Naive Bayes model using the prepared data.
    print(logprior)  # Prints the learned log prior.
    print(len(loglikelihood))  # Prints the number of words stored in the log-likelihood dictionary.

    my_utt = "To be or not to be, that is the question."  # Defines a sample utterance to classify.
    p = naive_bayes_predict(my_utt, logprior, loglikelihood)  # Computes the Naive Bayes score for the sample utterance.
    print("The expected output is", p)  # Prints the prediction score for the sample utterance.

    print(  # Starts printing the formatted Naive Bayes accuracy message.
        "Naive Bayes accuracy = %0.4f"  # Defines the output string format with four decimal places.
        % (test_naive_bayes(utts, ys, logprior, loglikelihood))  # Inserts the calculated accuracy into the formatted string.
    )  # Closes the print statement.