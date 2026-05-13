#!/usr/bin/env python
# coding: utf-8  # Declares that this Python file uses UTF-8 text encoding.

# In[4]:  # Indicates this code originally came from Jupyter Notebook cell number 4.


import re  # Imports Python's regular expression module for cleaning text.
import random  # Imports Python's random module for random word selection.
from nltk.tokenize import word_tokenize  # Imports NLTK's word tokenizer for splitting text into tokens.
from collections import defaultdict, deque  # Imports defaultdict for automatic lists and deque for efficient queue behavior.



class MarkovChain:  # Defines a MarkovChain class for building and generating text.
    def __init__(self):  # Defines the constructor that runs when a MarkovChain object is created.
        self.lookup_dict = defaultdict(list)  # Creates a dictionary where each key automatically maps to an empty list.
        self._seeded = False  # Tracks whether the random number generator has already been seeded.
        self.__seed_me()  # Calls the private method that seeds the random number generator.

    def __seed_me(self, rand_seed=None):  # Defines a private method to seed the random number generator.
        if self._seeded is not True:  # Checks whether seeding has not already happened.
            try:  # Starts a try block in case seeding fails.
                if rand_seed is not None:  # Checks whether a specific random seed was provided.
                    random.seed(rand_seed)  # Seeds the random module with the provided seed.
                else:  # Runs when no specific seed was provided.
                    random.seed()  # Seeds the random module using system time or another default source.
                self._seeded = True  # Marks the random number generator as successfully seeded.
            except NotImplementedError:  # Handles the rare case where random seeding is not implemented.
                self._seeded = False  # Keeps the seeded flag as false if seeding failed.

    def add_document(self, str):  # Defines a method that adds a text document to the Markov chain.
        preprocessed_list = self._preprocess(str)  # Cleans and tokenizes the input text.
        pairs = self.__generate_tuple_keys(preprocessed_list)  # Generates pairs of neighboring words.
        for pair in pairs:  # Loops through each word pair.
            self.lookup_dict[pair[0]].append(pair[1])  # Stores the second word as a possible follower of the first word.

    def _preprocess(self, str):  # Defines a helper method to clean and tokenize text.
        cleaned = re.sub(r"\W+", " ", str).lower()  # Replaces non-word characters with spaces and lowercases the text.
        tokenized = word_tokenize(cleaned)  # Splits the cleaned text into word tokens.
        return tokenized  # Returns the list of tokens.

    def __generate_tuple_keys(self, data):  # Defines a private generator method that creates adjacent word pairs.
        if len(data) < 1:  # Checks whether the token list is empty.
            return  # Stops the method if there is no data.

        for i in range(len(data) - 1):  # Loops through each token except the last one.
            yield [data[i], data[i + 1]]  # Yields a pair containing the current word and the next word.

    def generate_text(self, max_length=50):  # Defines a method that generates text up to a maximum length.
        context = deque()  # Creates a deque to hold the current word context.
        output = []  # Creates a list to store generated words.
        if len(self.lookup_dict) > 0:  # Checks whether the Markov chain has any learned word transitions.
            self.__seed_me(rand_seed=len(self.lookup_dict))  # Seeds randomness using the number of dictionary entries.
            chain_head = [list(self.lookup_dict)[0]]  # Selects the first dictionary key as the starting word.
            context.extend(chain_head)  # Adds the starting word to the context deque.

            while len(output) < (max_length - 1):  # Continues generating until the output reaches the requested length.
                next_choices = self.lookup_dict[context[-1]]  # Gets possible next words for the current last context word.
                if len(next_choices) > 0:  # Checks whether there are possible next words.
                    next_word = random.choice(next_choices)  # Randomly selects one possible next word.
                    context.append(next_word)  # Adds the selected word to the context.
                    output.append(context.popleft())  # Moves the oldest context word into the output.
                else:  # Runs if there are no possible next words.
                    break  # Stops text generation.
            output.extend(list(context))  # Adds any remaining context words to the output.
        return " ".join(output)  # Joins the generated words into a single string and returns it.


if __name__ == "__main__":  # Runs the following block only when this file is executed directly.
    with open(r"C:\Users\HADI\Downloads\hamlet.txt", "r", encoding="utf-8") as f:  # Opens the Hamlet text file in read mode with UTF-8 encoding.
        text = f.read()  # Reads the full file contents into a string.
    HMM = MarkovChain()  # Creates a new MarkovChain object.
    HMM.add_document(text)  # Adds the Hamlet text to the Markov chain.

    print(HMM.generate_text(max_length=25))  # Generates and prints 25 words of Markov-chain text.