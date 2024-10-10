# NLP Projects Repository

This repository contains three Natural Language Processing (NLP) small projects done during my NLP class. The projects focus on various aspects of tweet analysis and sentence similarity, with a specific emphasis on Algerian tweets, which include multiple languages such as Arabic, Berber, English, French, and Arabizi. Below are the descriptions of the three projects:

## Project 1: Tweets’ Similarity

**Objective**: Implement a program to detect similar tweets, focusing on Algerian tweets written in Arabic, Berber, English, French, and Arabizi. This project aims to identify spammers and bots that tend to repeat their tweets with minor modifications.

**Use Case**: Detecting spammers and robots through tweet similarity detection, which can help filter repetitive or automated content on social media platforms.

**Approach**: 
- Preprocess the tweets to handle multilingual text.
- Use similarity measures (such as cosine similarity) to compare the content of tweets.
- Identify patterns that suggest repetitive behavior or spam.

---

## Project 2: Tweets’ Grammaticality Judgment

**Objective**: Test the grammaticality of Algerian tweets by analyzing the structure of the sentences rather than their meaning. The goal is to statistically determine the likelihood of a tweet being grammatically correct based on N-Grams.

**Use Case**: This project helps assess how well tweets conform to the grammatical rules of a language, which can be useful for language processing tasks or even for detecting low-quality or poorly written content.

**Approach**:
- Use N-Grams to model the probability of tweet structures.
- Analyze tweet datasets to calculate the likelihood of grammatical correctness.
- Provide a probability score for each tweet's grammaticality.

---

## Project 3: Sentences’ Similarity Based on Their Words

**Objective**: Measure the similarity between sentences based on their words. Sentence similarity is useful in various NLP tasks such as Information Retrieval (IR), Automatic Text Summarization (ATS), Plagiarism Detection, and Machine Translation (MT).

**Use Case**: 
- Enhance Information Retrieval (IR) by considering sentence similarity in search queries.
- Reduce redundancy in text summarization by identifying similar sentences.
- Detect plagiarism by comparing sentences from different documents.
- Evaluate machine translation quality by comparing the similarity of translated sentences to reference translations.

**Approach**:
- Represent sentences as vectors using various techniques (e.g., word embeddings).
- Calculate similarity scores between sentences using vector similarity measures like cosine similarity.

---

## How to Use

Each project is located in its respective folder. No additional package is required, you just need to run the *.py* file. 

You can clone this repository using the following command:

```bash
git clone https://github.com/ChakibBrg/NLP_TPs.git
