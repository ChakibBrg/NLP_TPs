#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Students:
#     - ...
#     - ...

"""QUESTIONS/ANSWERS

----------------------------------------------------------
Q1: 
We want to consider unknown words in general texts, propose a solution.
We want to take in consideration different variants of Arabizi, propose a solution.

A1:
...
----------------------------------------------------------


----------------------------------------------------------
Q2:
If we train a model on the inverse of texts.
Will we get the same probability as the one in the right direction? Why?
Will this affect grammaticality judgment? Why?

A2:
...
----------------------------------------------------------


----------------------------------------------------------
Q3:
Can we use Viterbi to calculate the probability of a text? Why/How?

A3:
...
----------------------------------------------------------


----------------------------------------------------------
Q4:
Describe how can we decide that a text is grammatical or not based on its probability.

A4:
...
----------------------------------------------------------


"""

import math, json, os
from typing import Dict, List, Any, Tuple

# ====================================================
# ================= BiGram class =====================
# ====================================================
class BiGram:

    def __init__(self):
        self.uni_grams = {'<s>': 0, '</s>': 0}
        self.bi_grams = {}

    def fit(self, data:List[List[str]]) -> None:
        """Trains the current bi-gram model

        Args:
            data (List[List[str]]): a list of tokenized sentences.
        """

        for i in range(len(data)):
            seg = data[i]
            seg.append('</s>')
            past = '<s>'
            self.uni_grams['<s>'] += 1
            for j in range(len(seg)):
                unigram = seg[j]
                freq_uni = 1
                if unigram in self.uni_grams:
                    freq_uni = self.uni_grams[unigram] + 1
                self.uni_grams[unigram] = freq_uni

                bigram = past + ' ' + unigram
                freq_bi = 1
                if bigram in self.bi_grams:
                    freq_bi = self.bi_grams[bigram] + 1
                self.bi_grams[bigram] = freq_bi

                past = unigram

    # HINT: use math.log
    def score(self, past:str, current:str, alpha:float=1.) -> float:
        """Estimates the conditional probability P(current|past)

        Args:
            past (str): the past token.
            current (str): the current token.
            alpha (float, optional): Lidstone factor. Defaults to 1..

        Returns:
            float: conditional log-probability
        """
        freq_bi = alpha
        bigram = past + ' ' + current
        if bigram in self.bi_grams:
            freq_bi += self.bi_grams[bigram]
        freq_uni = alpha * len(self.uni_grams)
        if past in self.uni_grams:
            freq_uni += self.uni_grams[past]

        return math.log(freq_bi) - math.log(freq_uni)

    def predict(self, tokens:List[str], alpha:float=1.) -> float:
        """Predicts the log probbability of a sequence of tokens
        P(t1 ... tn)

        Args:
            tokens (List[str]): a list of tokens (without padding).
            alpha (float, optional): Lidstone's factor. Defaults to 1..

        Returns:
            float: Log probability of the sequence.
        """
        score = 0.
        past = '<s>'
        for token in tokens:
            score += self.score(past, token, alpha=alpha)
            past = token
        score += self.score(past, '</s>')
        return score

    # --------------- Implemented methods --------------
    def export_json(self) -> Dict[str, Any]:
        """Serialize the object as json object

        Returns:
            Dict[str, Any]: json representation of the current object.
        """
        return json.dumps(self.__dict__)

    def import_json(self, data:Dict[str, Any]):
        """Populate the current object using json serialization

        Args:
            data (Dict[str, Any]): json representation
        """
        for key in data:
            self.__dict__[key] = data[key]
            

# ====================================================
# ================= NGram class =====================
# ====================================================
# TODO Complete Ngram class (__init__, fit, score, predict)
class NGram:

    def __init__(self, N=2, alpha:float=1., Lambda=[]):
        
        # Exception
        if N < 1 or N > 6:
            raise ValueError("N must be between 1 and 6")
        
        self.N = N
        self.grams = {}
        self.alpha = alpha
        self.Lambda = Lambda
        # Add more code here
        self.V = 0

    def fit(self, data:List[List[str]]) -> None:
        flat_list = [word for sentence in data for word in sentence]
        self.V = len(set(flat_list))
        for sentence in data:
            for i in range(self.N):
                sentence.insert(0, '<s>')
                sentence.append('</s>')
            for j in range(len(sentence) - self.N + 1):
                ngram = sentence[j:j+self.N]
                if ngram.count('<s>') == self.N or ngram.count('</s>') == self.N:
                    continue
                ngram_str = " ".join(ngram)
                freq = 1
                if ngram_str in self.grams:
                    freq = self.grams[ngram_str] + 1
                self.grams[ngram_str] = freq

    def score(self, past:List[str], current:str, smooth=None) -> float:
        print(past, current)

        if smooth == 'lidstone' or self.N == 1:
            if self.N > 1:
                past_grams = ' '.join(past)
            else:
                past_grams = past[0]
            freq_past = 0
            if past_grams in self.grams:
                freq_past += self.grams[past_grams]

            freq_current = 0
            current_gram = past_grams + ' ' + current
            if current_gram in self.grams:
                freq_current += self.grams[current_gram]


            if freq_current + self.alpha == 0:
                nominateur = -math.inf
            else:
                nominateur = math.log(freq_current + self.alpha)

            if freq_past + self.alpha * self.V == 0:
                denominateur = -math.inf
            else: 
                denominateur = math.log(freq_past + self.alpha * self.V)

            return nominateur - denominateur
        elif smooth == 'interpolation':
            somme = 0
            for i in range(len(self.Lambda)):
                past_grams = " ".join(past[i:])
                freq_past = 0
                if past_grams in self.grams:
                    freq_past += self.grams[past_grams]
                freq_current = 0
                current_gram = past_grams + ' ' + current
                if current_gram in self.grams:
                    freq_current += self.grams[current_gram]

                if freq_current != 0 and freq_past != 0:
                    somme += self.Lambda[i] * (freq_current / freq_past)
            return somme
        
        else:
            past_grams = " ".join(past)
            print("past grams: ", past_grams)
            freq_past = 0
            if past_grams in self.grams:
                freq_past += self.grams[past_grams]
            freq_current = 0
            current_gram = past_grams + ' ' + current
            if current_gram in self.grams:
                freq_current += self.grams[current_gram]
            print("freq_current: ", freq_current)
            print("freq_past: ", freq_past)
            if freq_current == 0:
                nominateur = -math.inf
            else:
                nominateur = math.log(freq_current + self.alpha)

            if freq_past == 0:
                return -math.inf
            else: 
                denominateur = math.log(freq_past + self.alpha * self.V)

            return nominateur - denominateur
        

    def predict(self, tokens:List[str], smooth=None) -> float:
        score = 0.0
        for i in range(len(tokens) - self.N + 1):
            ngram = tokens[i:i+self.N]
            score += self.score(ngram[:-1], ngram[-1], smooth=smooth)
        return score





    # --------------- Implemented methods --------------
    def export_json(self) -> Dict[str, Any]:
        """Serialize the object as json object

        Returns:
            Dict[str, Any]: json representation of the current object.
        """
        return json.dumps(self.__dict__)

    def import_json(self, data:Dict[str, Any]):
        """Populate the current object using json serialization

        Args:
            data (Dict[str, Any]): json representation
        """
        for key in data:
            self.__dict__[key] = data[key]

# ====================================================
# ============== Grammaticality class =================
# ====================================================
class Grammaticality:
    def __init__(self, models:Dict[str, NGram]):
        self.models:Dict[str, NGram] = models

    def fit(self, url:str):
        text_words = []
        with open(url, 'r', encoding='utf8') as f:
            for line in f:
                if len(line) > 1:
                    line = line.strip(' \t\r\n')
                    words = line.split()
                    text_words.append(words) 

        for name in self.models:
            self.models[name].fit(text_words)
        

    def predict(self, text:str, smooth=None) -> Dict[str, bool]:
        prob = {}

        for name in self.models:
            # This is question 3
            prob[name] = self.models[name].predict(text.split(), smooth=smooth)

        # This is question 4
        return None


    def populate_model(self, url:str):
        """Fill the model from a json serialized object

        Args:
            url (str): URL of json file
        """
        f = open(url, 'r')
        data = json.load(f)
        self.word = data['word']
        self.models = {}
        for code in data['models']:
            self.models[code] = NGram()
            self.models[code].import_json(data['models'][code])
        f.close()

    def save_model(self, url:str):
        """Serialize the model into a json file

        Args:
            url (str): URL of the json file.
        """
        f = open(url, 'w')
        json.dump(self.__dict__, f, default=lambda o: o.export_json())
        f.close()

    # # private static method to load evaluation samples
    # def __get_evaluation(self, url:str) -> List[Tuple[str, str]]:
    #     result = []
    #     f = open(url, 'r')
    #     for l in f:
    #         l = l.strip(' \t\r\n')
    #         if len(l) < 2:
    #             continue
    #         info = l.split('#')
    #         result.append((info[0], info[1]))
    #     f.close()
    #     return result

    # def evaluate(self, url:str, alpha:float=1.) -> Dict[str, Any]:
    #     total = 0
    #     found = 0
    #     counts = {}# code: [true, pred, true.pred]
    #     for sent, code in self.__get_evaluation(url):
    #         pred = self.predict(sent, alpha=alpha)
    #         if code not in counts:
    #             counts[code] = [0, 0, 0]
    #         if pred not in counts:
    #             counts[pred] = [0, 0, 0]
    #         counts[code][0] += 1
    #         counts[pred][1] += 1
    #         if pred == code:
    #             found += 1
    #             counts[code][2] += 1
    #         total += 1
        
    #     res = {'accuracy': found/total}
    #     for code in counts:
    #         res[code] = {
    #             'R': 0.0 if counts[code][0] == 0 else counts[code][2]/counts[code][0],
    #             'P': 0.0 if counts[code][1] == 0 else counts[code][2]/counts[code][1],
    #         }
            
    #     return res


# ====================================================
# ===================== Tests ========================
# ====================================================

texts = [
    ['a', 'computer', 'can', 'help', 'you'],
    ['he', 'wants', 'to', 'help', 'you'],
    ['he', 'wants', 'a', 'computer'],
    ['he', 'can', 'swim']
]

fits = {
    'Unigram': {'a': 2,
                'computer': 2,
                'can': 2,
                'help': 2,
                'you': 2,
                'he': 3,
                'wants': 2,
                'to': 1,
                'swim': 1,},
    'Bigram': {'<s> a': 1,
                'a computer': 2,
                'computer can': 1,
                'can help': 1,
                'help you': 2,
                'you </s>': 2,
                '<s> he': 3,
                'he wants': 2,
                'wants to': 1,
                'to help': 1,
                'wants a': 1,
                'computer </s>': 1,
                'he can': 1,
                'can swim': 1,
                'swim </s>': 1,},
        'Trigram': {'<s> <s> a': 1,
                    '<s> a computer': 1,
                    'a computer can': 1,
                    'computer can help': 1,
                    'can help you': 1,
                    'help you </s>': 2,
                    'you </s> </s>': 2,
                    '<s> <s> he': 3,
                    '<s> he wants': 2,
                    'wants to help': 1,
                    'to help you': 1,
                    'he wants a': 1,
                    'wants a computer': 1,
                    'a computer </s>': 1,
                    'computer </s> </s>': 1,
                    '<s> he can': 1,
                    'he can swim': 1,
                    'can swim </s>': 1,
                    'swim </s> </s>': 1,}
}

tests = [
    ['he', 'can', 'help', 'you'],
    ['he', 'wants', 'to', 'swim'],
    ['he', 'can', 'help', 'us'],
]

scores = [
    {'Unigram': (math.log(2/17), math.log(3/26), math.log(3/26)), 
     'Bigram': (0.0, math.log(3/13), math.log(0.7 + 0.6/17)),
     'Trigram': (0.0, math.log(2/13), math.log(0.55 + 0.4/17)),
    },
    {'Unigram': (math.log(1/17), math.log(2/26), math.log(2/26)), 
     'Bigram': (-math.inf, math.log(1/12), math.log(0.3/17)),
     'Trigram': (-math.inf, math.log(1/12), math.log(0.3/17)),
    },
    {'Unigram': (-math.inf, math.log(1/26), math.log(1/26)), 
     'Bigram': (-math.inf, math.log(1/11), -math.inf),
     'Trigram': (-math.inf, math.log(1/10), -math.inf),
    },
]

predicts = [
    {'Unigram': (math.log(3/17) + 3 * math.log(2/17), 
                 math.log(4/26) + 2 * math.log(3/26), 
                 math.log(4/26) + 3 * math.log(3/26)), 
     'Bigram': (math.log(3/4) + math.log(1/3) + math.log(1/2), 
                math.log(4/15) + math.log(2/14) + math.log(2/13) + 2*math.log(3/13), 
                math.log(2.1/4 + 0.9/17) + math.log(0.7/3+0.6/17) + math.log(0.7/2+0.6/17) + 2*math.log(0.7+0.6/17)),
     'Trigram': (-math.inf,
                 math.log(4/15) + math.log(2/14) + math.log(2/12) + 2*math.log(3/13), 
                 -math.inf),   
    },
    {'Unigram': (math.log(3/17) + math.log(2/17) + 2 * math.log(1/17), 
                 math.log(4/26) + math.log(3/26) + 2 * math.log(2/26),  
                 math.log(4/26) + math.log(3/26) + 2 * math.log(2/26)), 
     'Bigram': (-math.inf, 
                math.log(4/15) + math.log(3/14) + math.log(2/13) + math.log(1/12) + math.log(2/15), 
                math.log(0.9/17) + math.log(0.6/17) + 2 * math.log(0.3/17)),
     'Trigram': (-math.inf, 
                 'NC', 
                 math.log(0.6/17) + math.log(0.4/17) + 2 * math.log(0.2/17)),
    },
    {'Unigram': (-math.inf, 'NC', 'NC'), 
     'Bigram': (-math.inf, 'NC', -math.inf),
     'Trigram': (-math.inf, 'NC', -math.inf),
    },
]



def test_new():
    for N in range(-1, 2):
        try:
            gram = NGram(N=N)
            print('it works for N=', N)
        except Exception as e:
            print('Exception for N=', N)
            print('message: ', str(e))


def test_gram_fit():

    unigram = NGram(N=1)
    unigram.fit(texts)

    bigram = NGram(N=2)
    bigram.fit(texts)

    trigram = NGram(N=3)
    trigram.fit(texts)

    print('========= NGram fit test =========')
    print('     Order is not important')
    print('     your unigrams', unigram.grams)
    print('     must be', fits['Unigram'], "\n")
    print('     your bigrams', bigram.grams)
    print('     must be', fits['Bigram'], "\n")
    print('     your trigrams', trigram.grams)
    print('     must be', fits['Trigram'])


def test_ngram_score():
    models = [
        # ('Unigram', NGram(N=1), 1),
        ('Bigram', NGram(N=2, alpha=1., Lambda=[0.3, 0.7]), 2),
        ('Trigram', NGram(N=3, alpha=1., Lambda=[0.2, 0.3, 0.5]), 3),
        # ('Bigram2', NGram(N=2, alpha=0.001, Lambda=[0.7, 0.3]), 2),
        # ('Trigram2', NGram(N=3, alpha=0.001, Lambda=[0.5, 0.3, 0.2]), 3)
    ]

    print('========= Score test =========')
    for name, mdl, N in models:
        print('---------------------------')
        mdl.fit(texts)
        print('model:', name)
        for i, test in enumerate(tests):
            print('log p(', test[-1], '|', ' '.join(test[-N:-1]), ')= ', scores[i][name][0], ', yours=', mdl.score(test[-N:-1], test[-1], smooth=None))
            print('log p_L(', test[-1], '|', ' '.join(test[-N:-1]), ')= ', scores[i][name][1], ', yours=', mdl.score(test[-N:-1], test[-1], smooth='lidstone'))
            print('log p_I(', test[-1], '|', ' '.join(test[-N:-1]), ')= ', scores[i][name][2], ', yours=', mdl.score(test[-N:-1], test[-1], smooth='interpolation'))


def test_ngram_predict():
    models = [
        ('Unigram', NGram(N=1), 1),
        ('Bigram', NGram(N=2, alpha=1., Lambda=[0.3, 0.7]), 2),
        ('Trigram', NGram(N=3, alpha=1., Lambda=[0.2, 0.3, 0.5]), 3),
        # ('Bigram2', NGram(N=2, alpha=0.001, Lambda=[0.7, 0.3]), 2),
        # ('Trigram2', NGram(N=3, alpha=0.001, Lambda=[0.5, 0.3, 0.2]), 3)
    ]

    print('========= Predict test =========')
    for name, mdl, N in models:
        print('---------------------------')
        mdl.fit(texts)
        print('model:', name)
        for i, test in enumerate(tests):
            print('log p_L(', test, ')= ', predicts[i][name][0], ', yours=', mdl.prdict(test, smooth='lidstone'))
            print('log p_I(', test, ')= ', predicts[i][name][1], ', yours=', mdl.prdict(test, smooth='interpolation'))


def test_grammaticality():
    gramm = Grammaticality()
#     program_word = LangDetector(word=True)

#     program_char.fit('data')
#     program_word.fit('data')

#     # program_char.save_model('./char_based.json')
#     # program_word.save_model('./word_based.json')

#     print('------- char ----------')
#     print(program_char.evaluate('data/lang.eval'))
#     print('------- word ----------')
#     print(program_word.evaluate('data/lang.eval'))


# ====================================================
# =================== MAIN PROGRAM ===================
# ====================================================

# Activate unitary tests one by one and desactivate the rest.
if __name__ == '__main__':
    # test_new()
    test_gram_fit()
    test_ngram_score()
    # test_ngram_predict()
    # test_grammaticality()

