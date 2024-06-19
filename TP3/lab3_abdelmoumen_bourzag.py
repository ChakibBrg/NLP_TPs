#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Students:
#     - Hatem Mohamed ABDELMOUMEN
#     - Mohamed Chakib BOURZAG

"""QUESTIONS/ANSWERS

----------------------------------------------------------
Q1: Why the two sentences "cats chase mice" and "mice chase cats" are considered similar using all words and sentences encoding?
	Propose a solution to fix this.

A1:

The issue : Using CentroidSentRep, both sentences will be transformed into a centroid vector representing the average of the word vectors. Since "cats", "chase", and "mice" are the only words present in both sentences, the centroid vector will be the same for both sentences (average of the same vectors), making them appear identical in the vector space.
Solutions :
1. Position-aware centroids: Weight the word vectors by their positions in the sentence. Words at the beginning and end of the sentence can have different weights.
2. N-gram based representation: Instead of using individual words, we use n-grams to capture more context. For example, "cats chase" and "chase mice" vs. "mice chase" and "chase cats".
3. RNNs or Transformers: These models naturally capture word order. We can train an RNN or a Transformer model to get sentence embeddings that take into account the order of words.


----------------------------------------------------------

----------------------------------------------------------
Q2:  Why using concatenation, the 2nd and 4th sentences are similar?
	Can we enhance this method, so they will not be as similar?

A2: 

The issue : When using concatenation, the first few words in both sentences will be the same, and only the additional words at the end of the longer sentence will differ. If the max_words parameter isn't large enough, the additional words are deleted, the longer sentence "a computer can help you and you can help" is truncated to match the length of the shorter sentence "a computer can help you". As a result, both sentences end up identical, thus having the same representation.
Solutions :
1. Adjustable max_words: Instead of fixing the maximum number of words, dynamically adjust the maximum length based on the length of the longest sentence. This way, Instead of truncating the longer sentence, we can dynamically pad the shorter sentence with zeros to match the length of the longer one. Therefore, both sentences retain their original lengths, and their differences are preserved.
2. Sentence summarization: Instead of simple truncation or padding, use an NLP model to summarize or extract the main part of the sentence. This way, redundant or less important information is discarded, and we ensure that only the core meaning of each sentence contributes to the similarity calculation.


----------------------------------------------------------

----------------------------------------------------------
Q3: compare between the two sentence representation, indicating their limits.

A3: 

General Comparison :
- ConcatSentRep: This method represents a sentence by concatenating the vector representations of its words up to a maximum length, truncating or padding as necessary.
- CentroidSentRep: This method represents a sentence by computing the centroid (average) of the vector representations of its words.


Here's a summarizing table for the pros and cons of each representation :

|--------------------|-------------------------------------|----------------------------------------------------------|
| Representation     | Pros                                | Cons                                                     |
|--------------------|-------------------------------------|----------------------------------------------------------|
| ConcatSentRep      | - Flexible, adjustable max length   | - Limited by fixed size                                  |
|                    | - Maintains some word order         | - Truncation or padding can lead to loss of information  |
|--------------------|-------------------------------------|----------------------------------------------------------|
| CentroidSentRep    | - Simple                            | - Ignores word order                                     |
|                    | - Handles varying sentence lengths  | - Loss of important distinctions                         |
|                    |                                     | - Dilution of information                                |
|                    |                                     | - May oversimplify meaning                               |
|--------------------|-------------------------------------|----------------------------------------------------------|


Here's a summarizing table for the limits of these representations, highlighting how it's considered as a limit :


|------------------------|------------------------------------------------------|------------------------------------------------|
| Limits                 | ConcatSentRep                                        | CentroidSentRep                                |
|------------------------|------------------------------------------------------|------------------------------------------------|
| Information loss       | Truncation/padding                                   | Averaging                                      |
| Word order             | Partially maintained (only up to a certain length)   | Ignored completely                             |
| Sentence length        | Cutting off sentence meaning (truncation)            | Generalization and loss of specific details    |
|------------------------|------------------------------------------------------|------------------------------------------------|




----------------------------------------------------------

"""

import re
import os
import sys
import math
import json
import random
from functools import reduce
from typing import Dict, List, Tuple, Set, Union, Any

# ====================================================
# ============== Usefull functions ===================
# ====================================================

def vec_plus(X: List[float], Y: List[float]) -> List[float]:
    """Given two lists, we calculate the vector sum between them

    Args:
        X (List[float]): The first vector
        Y (List[float]): The second vector

    Returns:
        List[float]: The vector sum (element-wize sum)
    """
    return list(map(sum, zip(X, Y)))

def vec_divs(X: List[float], s: float) -> List[float]:
    """Given a list and a scalar, it returns another list
       where the elements are divided by the scalar

    Args:
        X (List[float]): The list to be divided
        s (float): The scalar

    Returns:
        List[float]: The resulted div list.
    """
    return list(map(lambda e: e/s, X))

# ====================================================
# ====== API: Application Programming Interface ======
# ====================================================

class WordRep:

    def fit(self, text: List[List[str]]) -> None:
        raise 'Not implemented, must be overriden'

    def transform(self, words: List[str]) -> List[List[float]]:
        raise 'Not implemented, must be overriden'

class SentRep:
    def __init__(self, word_rep: WordRep) -> None:
        self.word_rep = word_rep

    def transform(self, text: List[List[str]]) -> List[List[float]]:
        raise 'Not implemented, must be overriden'

class Sim:
    def __init__(self, sent_rep: SentRep) -> None:
        self.sent_rep = sent_rep

    def calculate(self, s1: List[str], s2: List[str]) -> float:
        raise 'Not implemented, must be overriden'
    

# ====================================================
# =========== WordRep implementations ================
# ====================================================

class OneHot(WordRep):
    def __init__(self, specials: List[str] = []) -> None:
        super().__init__()
        self.specials = specials
        self.word_to_index = {}
        self.index_to_word = {}

    def fit(self, text: List[List[str]]) -> None:
        # Add "[UNK]" as the first token
        if "[UNK]" not in self.word_to_index:
            self.word_to_index["[UNK]"] = len(self.word_to_index)
            self.index_to_word[len(self.index_to_word)] = "[UNK]"

        # Add the given special tokens after "[UNK]"
        for special_token in self.specials:
            if special_token not in self.word_to_index:
                self.word_to_index[special_token] = len(self.word_to_index)
                self.index_to_word[len(self.index_to_word)] = special_token

        # Add words from the text to the one-hot representation
        for sentence in text:
            for word in sentence:
                if word not in self.word_to_index:
                    self.word_to_index[word] = len(self.word_to_index)
                    self.index_to_word[len(self.index_to_word)] = word

    def transform(self, words: List[str]) -> List[List[float]]:
        # Initialize the sentence representation with zeros
        sentence_representation = [[0] * len(self.word_to_index) for _ in range(len(words))]
        # Encode each word in the sentence as one-hot
        for i, word in enumerate(words):
            if word in self.word_to_index:
                index = self.word_to_index[word]
                sentence_representation[i][index] = 1
            else:
                # Use the [UNK] token if the word is not found
                unk_index = self.word_to_index.get("[UNK]", None)
                if unk_index is not None:
                    sentence_representation[i][unk_index] = 1
        return sentence_representation
        

class TermTerm(WordRep):
    def __init__(self, window=2) -> None:
        super().__init__()
        self.window = window
        self.term_matrix = {}

    def fit(self, text: List[List[str]]) -> None:
        # Build the term-term representation from the text
        for sentence in text:
            for i, word in enumerate(sentence):
                if word not in self.term_matrix:
                    self.term_matrix[word] = {}
                # Iterate through words within the window around the current word
                for j in range(max(0, i - self.window), min(len(sentence), i + self.window + 1)):
                    if i != j:
                        context_word = sentence[j]
                        if context_word not in self.term_matrix[word]:
                            self.term_matrix[word][context_word] = 0
                        self.term_matrix[word][context_word] += 1

    def transform(self, words: List[str]) -> List[List[float]]:
        # Initialize the sentence representation with zeros
        sentence_representation = [[0] * len(self.term_matrix) for _ in range(len(words))]
        # Iterate over each word in the sentence
        for i, word in enumerate(words):
            # Check if the word has recorded contexts
            if word in self.term_matrix:
                context_counts = self.term_matrix[word]
                # Iterate over each context of the word
                for context_word, count in context_counts.items():
                    # Find the index of the context in the sentence representation
                    if context_word in self.term_matrix:
                        context_index = list(self.term_matrix.keys()).index(context_word)
                        # Add the context weight to the sentence representation
                        sentence_representation[i][context_index] = count
        return sentence_representation
        
    
# ====================================================
# =========== SentRep implementations =================
# ====================================================

class ConcatSentRep(SentRep):
    def __init__(self, word_rep: WordRep, max_words: int = 10) -> None:
        super().__init__(word_rep)
        self.max_words = max_words

    def transform(self, text: List[List[str]]) -> List[List[float]]:
        sentences_representation = []
        for sentence in text:
            word_vectors = self.word_rep.transform(sentence)
            # Truncate or pad the word vectors
            if len(word_vectors) > self.max_words:
                word_vectors = word_vectors[:self.max_words]
            elif len(word_vectors) < self.max_words:
                padding = [[0.0] * len(word_vectors[0]) for _ in range(self.max_words - len(word_vectors))]
                word_vectors.extend(padding)
            # Flatten the list of word vectors into a single vector
            flattened_vector = [value for sublist in word_vectors for value in sublist]
            sentences_representation.append(flattened_vector)
        return sentences_representation


class CentroidSentRep(SentRep):
    def __init__(self, word_rep: WordRep) -> None:
        super().__init__(word_rep)

    def transform(self, text: List[List[str]]) -> List[List[float]]:
        sentences_representation = []
        for sentence in text:
            word_vectors = self.word_rep.transform(sentence)
            if not word_vectors:
                # Handle empty word vectors by appending a zero vector
                sentences_representation.append([0.0] * len(self.word_rep.transform([""])[0]))
                continue
            # Calculate the centroid of the word vectors
            centroid = [0.0] * len(word_vectors[0])
            for vector in word_vectors:
                centroid = vec_plus(centroid, vector)
            centroid = vec_divs(centroid, len(word_vectors))
            sentences_representation.append(centroid)
        return sentences_representation


# ====================================================
# =========== Sim implementations ================
# ====================================================

class EuclideanSim(Sim):
    def __init__(self, sent_rep: SentRep) -> None:
        super().__init__(sent_rep)

    def calculate(self, s1: List[str], s2: List[str]) -> float:
        # Transform sentences into vectors
        vec1 = self.sent_rep.transform([s1])[0]
        vec2 = self.sent_rep.transform([s2])[0]

        # Calculate the Euclidean distance between the two vectors
        diff_vector = vec_plus(vec1, [-x for x in vec2])
        distance = math.sqrt(sum(x ** 2 for x in diff_vector))

        # Calculate similarity based on the Euclidean distance
        similarity = 1 / (1 + distance)
        return similarity




# # ====================================================
# # ============ SentenceComparator class ==============
# # ====================================================

# NOT important

# ====================================================
# ===================== Tests ========================
# ====================================================

train_data = [
    ['a', 'computer', 'can', 'help', 'you'],
    ['he', 'can', 'help', 'you', 'and', 'he', 'wants', 'to', 'help', 'you'],
    ['he', 'wants', 'a', 'computer', 'and', 'a', 'computer', 'for', 'you']
]

test_data_wd = ['[S]', 'a', 'computer', 'wants', 'to', 'be', 'you']

test_data_st = [
    ['a', 'computer', 'can', 'help'],
    ['a', 'computer', 'can', 'help', 'you'],
    ['you', 'can', 'help', 'a', 'computer'],
    ['a', 'computer', 'can', 'help', 'you', 'and', 'you', 'can', 'help']
]

class DummyWordRep(WordRep):
    def __init__(self) -> None:
        super().__init__()
        self.code = {
            'a':        [1., 0., 0.],
            'computer': [0., 1., 0.],
            'can':      [0., 0., 1.],
            'help':     [1., 1., 0.],
            'you' :     [1., 0., 1.],
            'he':       [0., 1., 1.],
            'and':      [2., 0., 0.],
            'wants':    [0., 2., 0.],
            'to':       [0., 0., 2.],
            'for':      [2., 1., 0.]
        }

    def transform(self, words: List[str]) -> List[List[float]]:
        res = []
        for word in words:
            res.append(self.code[word])
        return res

def test_OneHot():
    oh_ref  = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
               [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
               [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]
    ohs_ref = [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
               [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
               [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]
    
    oh_enc = OneHot()
    ohs_enc = OneHot(specials=['[CLS]', '[S]'])

    oh_enc.fit(train_data)
    ohs_enc.fit(train_data)

    print('=========================================')
    print('OneHot class test')
    print('=========================================')
    oh_res = oh_enc.transform(test_data_wd)
    print('oneHot without specials')
    print(oh_res)
    print('should be')
    print(oh_ref)

    ohs_res = ohs_enc.transform(test_data_wd)
    print('oneHot with specials')
    print(ohs_res)
    print('should be')
    print(ohs_ref)

def test_TermTerm():
    tt2_ref = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
               [0, 4, 1, 0, 0, 1, 2, 1, 0, 1], 
               [4, 0, 1, 1, 1, 0, 2, 1, 0, 1], 
               [1, 1, 0, 1, 0, 2, 1, 0, 1, 0], 
               [0, 0, 0, 1, 1, 1, 0, 1, 0, 0], 
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
               [0, 1, 2, 3, 0, 1, 1, 0, 1, 1]]
    tt3_ref = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
               [2, 4, 1, 1, 1, 1, 2, 1, 0, 1], 
               [4, 2, 1, 1, 2, 1, 2, 1, 0, 1], 
               [1, 1, 0, 1, 2, 2, 2, 0, 1, 0], 
               [0, 0, 0, 1, 1, 1, 1, 1, 0, 0], 
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
               [1, 2, 2, 3, 0, 2, 1, 2, 1, 1]]

    tt2_enc = TermTerm()
    tt3_enc = TermTerm(window=3)

    tt2_enc.fit(train_data)
    tt3_enc.fit(train_data)

    print('=========================================')
    print('TermTerm class test')
    print('=========================================')
    tt2_res = tt2_enc.transform(test_data_wd)
    print('TermTerm window=2')
    print(tt2_res)
    print('should be')
    print(tt2_ref)

    tt3_res = tt3_enc.transform(test_data_wd)
    print('TermTerm window=3')
    print(tt3_res)
    print('should be')
    print(tt3_ref)

def test_ConcatSentRep():
    test = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], 
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0], 
            [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]]
    dummy_wd = DummyWordRep()
    concat_sent = ConcatSentRep(dummy_wd, max_words=5)
    print('=========================================')
    print('ConcatSentRep class test')
    print('=========================================')

    print(concat_sent.transform(test_data_st))
    print('must be')
    print(test)


def test_CentroidSentRep():
    test = [[0.5, 0.5, 0.25], 
            [0.6, 0.4, 0.4], 
            [0.6, 0.4, 0.4], 
            [0.7777777777777778, 0.3333333333333333, 0.4444444444444444]]
    dummy_wd = DummyWordRep()
    centroid_sent = CentroidSentRep(dummy_wd)
    print('=========================================')
    print('CentroidSentRep class test')
    print('=========================================')

    print(centroid_sent.transform(test_data_st))
    print('must be')
    print(test)

def test_Sim():
    test = [[0.5, 0.5, 0.25], 
            [0.6, 0.4, 0.4], 
            [0.6, 0.4, 0.4], 
            [0.7777777777777778, 0.3333333333333333, 0.4444444444444444]]
    dummy_wd = DummyWordRep()
    centroid_sent = CentroidSentRep(dummy_wd)
    sim = EuclideanSim(centroid_sent)
    print('=========================================')
    print('EuclideanSim class test')
    print('=========================================')

    print(centroid_sent.transform(test_data_st))
    print('must be')
    print(test)

# TODO: activate one test at once 
if __name__ == '__main__':
    test_OneHot()
    print()
    test_TermTerm()
    print()
    test_ConcatSentRep()
    print()
    test_CentroidSentRep()
    print()
    test_Sim()