import re
import os
import sys

from typing import Tuple, List

# Submit just this file, no zip, no additional files
# -------------------------------------------------

# Students:
#     - Abdelmoumen Hatem Mohamed
#     - Bourzag Mohamed Chakib

"""QUESTIONS/ANSWERS

----------------------------------------------------------
Q1: What are the problem(s) with normalization in our case (Algerian tweets)?

A1: There are many problems as:
    - Using multiple languages (Arabic, French, English and Berber) in a single tweet so it has a big language diversity
    so a normalization for a single language won't be enough here.
    - Abbreviations specific to the Algerian dialect: Algérie->dz, khoya->kho...
    - Some regions' dialect differences, for example between Arabic and Berber, east/west...
    - Using emojis as shown in the txt file (general problem).
    - Misspelling some words, or more precisely not having a standard spelling, for example: (tewelt, twlt), (kho, 5o)...

----------------------------------------------------------

----------------------------------------------------------
Q2: Why word similarity is based on edit distance and not vectorization such as TF?

A2: Because edit distance has the ability to capture structural similarity (to define whether to apply insertion, deletion or substitution)
    as it's based on character sequence and it's effective to detect morphological differences (which is so frequent in languages used in Algeria).
    Whereas, vectorization as TF is used to detect document similarity, information retrieval, and text classification, where the focus is on comparing 
    documents or larger text units based on the frequency of words (or their features), without caring about structural and morphological changes 
    explicitely which is important in our case here.

----------------------------------------------------------

----------------------------------------------------------
Q3: Why tweets similarity is proposed as such? 
    (not another formula such as the sum of similarity of the first tweet's words with the second's divided by max length)

A3: Because if we do as proposed between () we'll only consider the most similar words from T2 with respect to T1, and therfore we'll negligate the most similar words
    from T1 with respect to T2.
    For example; let's suppose we have T1=[w1, w2, w3] and T2=[t1, t2]:
    For each word of T1 let's suppose that they are all most similar to t2, and therfore by calculating as proposed between () t1 would never appear
    That's why we have to consider both sides and devide by the sum of both lenghts to get full information about similarities of all words in the tweets

----------------------------------------------------------


----------------------------------------------------------
Q4: Why blanks are being duplicated before using regular expressions in our case?

A4:     Because of the regular expressions ending with this [\s|$], so when it's not the end of a sentence but only passing to another word, this regular expression
        will detect the space and so it will delete it and concinait the resulting word with the next one, that's why we need to duplicate space
        so that after deleting the first space the two words stay separated
        
        For example; if we have this part of sentence "thanks dear reader", and according to our rules, the 's'
        as well as the 'k' will be gone; and to apply it, we need to have the end of the word so it will be
        used in the regex, so if the spaces are not duplicated, we will have as a result 'thandear reader'

----------------------------------------------------------

"""


# TODO Complete words similarity function
def word_sim(w1:str, w2:str) -> float:
    """Calculates Levenstein-based similarity between two words. 
    The function's words are interchangeable; i.e. levenstein(w1, w2) = levenstein(w2, w1)

    Args:
        w1 (str): First word.
        w2 (str): Second word.

    Returns:
        float: similarity.
    """

    if len(w1) * len(w2) == 0:
        return 0.0 # If one of them is empty then the distance is the length of the other

    D = []
    D.append([i for i in range(len(w2) + 1)])
    for i in range(len(w1)):
        l = [i+1]
        for j in range(len(w2)):
            s = D[i][j] + (0 if w1[i] == w2[j] else 1)
            m = min([s, D[i][j+1] + 1, l[j] + 1])
            l.append(m)
        D.append(l)

    return (max(len(w1), len(w2)) - D[len(w1)][len(w2)]) / max(len(w1), len(w2))


TASHKIIL	= [u'ِ', u'ُ', u'َ', u'ْ']
TANWIIN		= [u'ٍ', u'ٌ', u'ً']
OTHER       = [u'ـ', u'ّ']

# TODO Complete text normalization function
def normalize_text(text: str) -> str :
    """Normalize a text

    Args:
        text (str): source text

    Returns:
        str: result text
    """
    result = text.replace(' ', '  ') # duplicate the space
    result = re.sub('['+''.join(TASHKIIL+TANWIIN+OTHER)+']', '', result)
    result = result.lower()



    # SPECIAL =============================

    result = re.sub(r'[a-zA-Z0-9._]+@[a-zA-Z0-9._]+\.[a-zA-Z]+', '[MAIL]', result)
    result = re.sub(r'@\w+', '[USER]', result) 
    result = re.sub(r'#[^\s]+', '[HASH]', result) 
    result = re.sub(r'https://t\.co/\w+', '[LINK]', result) 



    # FRENCH/ENGLISH/BERBER ===============

    # Replacing a and e
    result = re.sub(r'é|è|ê|ë', 'e', result)
    result = re.sub(r'à|â|ä', 'a', result)

    # Ending s deletion
    result = re.sub(r'(\w+)(s)(\s|$)', r'\1 ', result)

    # French suffixes
    result = re.sub(r'(\w+)(ir|er|ement|ien|iens|euse|euses|eux)(\s|$)', r'\1 ', result)
    
    # English suffixes
    result = re.sub(r'(\w+[^(al)])(ly|al)(\s|$)', r'\1 ', result)
    result = re.sub(r'(\w+)(ally\s)', r'\1al ', result)

    # Berber suffixes
    result = re.sub(r'(\w+)(yas?|en)(\s|$)', r'\1 ', result)

    # English contractions
    result = re.sub(r'(\w+t)(\'s)', r'\1 is ', result)
    result = re.sub(r'(\w+)(n\'t)', r'\1 not ', result)

    # French contractions
    result = re.sub(r'(t|d|qu|l|s|j)(\')', r'\1e ', result)
    result = re.sub(r'(.+)(\')(.+)', r'\1e\3 ', result)



    # DZ ARABIZI ==========================

    # Negation
    result = re.sub(r'(ma)(\w+)(ch)(\s|$)', r'\2 ', result)

    # Suffixes
    result = re.sub(r'(\w+[^e])(ek|km|k)(\s|$)', r'\1 ', result)
    result = re.sub(r'(\w{2,})(a|i|ou?)(\s|$)', r'\1 ', result)
    result = re.sub(r'(\w{2,})(ha?)(\s|$)', r'\1 ', result)


    
    # ARABIC/DZ-ARABIC ====================

    # Negation
    result = re.sub(r'(^|\s)(م|ما)(\w+)(ش)(\s|$)' , r'ما \3 ' , result)
    result = re.sub(r'(^|\s)(\w+)(ش)(\s|$)' , r'ما \2 ' , result)


    # AL qualifiers
    result = re.sub(r'(ال|لل|فال|فل|وال|ول|بل|بال)(\w{2,})(\s|$)', r'\2 ', result)
    
    # Plural
    result = re.sub(r'(\w+)(ين|ون|ات|ال)(\s|$)', r'\1 ', result)

    # Prounouns
    result = re.sub(r'(\w{2,})(ني|ك|ه|ها|نا|كما|كم|كن|هما|هم|هن|وا)(\s|$)', r'\1 ', result)
  
    # Suffix
    result = re.sub(r'(\w{2,})(ة|ي|ا|و)(\s|$)', r'\1 ', result)


    # Final result ==========================

    return re.sub(r'[./:,;?!؟…]', ' ', result)


#=============================================================================
#                         IMPLEMANTED FUNCTIONS
#=============================================================================

def get_similar_word(word:str, other_words:List[str]) -> Tuple[str, float]:
    """Get the most similar word with its similarity

    Args:
        word (str): a word
        other_words (List[str]): list of target words

    Returns:
        Tuple[str, float]: the most similar word from the target + its similarity 
    """

    mx_sim = 0.
    sim_word = ''
    for oword in other_words:
        sim = word_sim(word, oword)
        if sim > mx_sim:
            mx_sim = sim 
            sim_word = oword

    return sim_word, mx_sim


def tweet_sim(tweet1:List[str], tweet2:List[str]) -> float: 
    """Similarity between two tweets

    Args:
        tweet1 (List[str]): tokenized tweet 1
        tweet2 (List[str]): tokenized tweet 2

    Returns:
        float: their similarity
    """
    sim = 0.
    for word in tweet1:
        sim += get_similar_word(word, tweet2)[1]
    
    for word in tweet2:
        sim += get_similar_word(word, tweet1)[1] 

    return sim/(len(tweet1) + len(tweet2))


def get_tweets(url:str='DZtweets.txt') -> List[List[str]]:
    """Get tweets from a file, where each tweet is in a line

    Args:
        url (str, optional): the URL of tweets file. Defaults to 'DZtweets.txt'.

    Returns:
        List[List[str]]: A list of tokenized tweets
    """
    result = []
    with open(url, 'r', encoding='utf8') as f:
        for line in f:
            if len(line) > 1:
                line = normalize_text(line)
                tweet = line.split()
                result.append(tweet)
    return result


#=============================================================================
#                             TESTS
#=============================================================================

def _word_sim_test():
    tests = [
        ('amine', 'immature', 0.25),
        ('immature', 'amine', 0.25),
        ('', 'immature', 0.0),
        ('amine', '', 0.0),
        ('amine', 'amine', 1.0),
        ('amine', 'anine', 0.8),
        ('amine', 'anine', 0.8),
    ]
    
    for test in tests:
        sim = word_sim(test[0], test[1])
        print('-----------------------------------')
        print('similarity between ', test[0], ' and ', test[1])
        print('yours ', sim, ' must be ', test[2])


def _normalize_text_test():
    tests = [
        ('@adlenmeddi @faridalilatfr Est-il en vente a Alger?', 
         ['[USER]', '[USER]', 'est-il', 'en', 'vente', 'a', 'alger']),
        ('@Abderra51844745 @officialPACCI @AfcfT @UNDP Many thanks dear friend', 
         ['[USER]', '[USER]', '[USER]', '[USER]', 'many', 'than', 'dear', 'friend']),
        ('Info@shahanaquazi.com ; I love your profile.', 
         ['[MAIL]', 'i', 'love', 'your', 'profile']),
        ('âme à périt éclairées fète f.a.t.i.g.u.é.é', 
         ['ame', 'a', 'perit', 'eclairee', 'fete', 'f', 'a', 't', 'i', 'g', 'u', 'e', 'e']),
        ('palestiniens Manchester dangereuses dangereux écouter complètement vetements', 
         ['palestin', 'manchest', 'danger', 'danger', 'ecout', 'complet', 'vet']),
        ('reading followers naturally emotional traditions notably', 
         ['read', 'follow', 'natural', 'emotion', 'tradition', 'notab']),
        ('iggarzen Arnuyas', 
         ['iggarz', 'arnu']),
        ("it's That's don't doesn't", 
         ['it', 'is', 'that', 'is', 'do', 'not', 'does', 'not']),
        ("l'éventail s'abstenir qu'ont t'avoir j'ai D'or D'hier t'en l'aïd p'tit", 
         ['le', 'eventail', 'se', 'absten', 'que', 'ont', 'te', 'av', 'je', 'ai', 'de', 'or', 'de', 'hi', "t'", 'le', 'aïd', 'petit']),
        ('mal9itch mata3rfch Bsahtek ywaf9ek ya3tik 3ndk', 
         ['l9it', 'ta3rf', 'bsaht', 'ywaf9', 'ya3t', '3nd']),
        ('Khaltiha Khaltih yetfarjou fhamto mousiba wladi  Chawala khmouss', 
         ['khalti', 'khalti', 'yetfarj', 'fhamt', 'mousib', 'wlad', 'chawal', 'khmous']),
        ('لَا حـــــــــــــــــــــوْلَ وَلَا قُوَّةَ إِلَّا بِاللَّهِ الْعَزِيزُ الْحَكِيمُ،', 
         ['لا', 'حول', 'ولا', 'قوة', 'إلا', 'له', 'عزيز', 'حكيم،']),
        ('منلبسوش ميخرجش ميهمناش مايهمنيش قستيهاش فهمتش معليش', 
         ['ما', 'نلبس', 'ما', 'يخرج', 'ما', 'يهم', 'ما', 'يهم', 'ما', 'قستي', 'ما', 'فهمت', 'ما', 'عل']),
        ('الطاسيلي للاحباب اللهم المورال الاتحادبات المصلحين والتنازلات الجزائري فالناس للسونترال بروفايلات والصومال', 
         ['طاسيل', 'احباب', 'لهم', 'مور', 'اتحادب', 'مصلح', 'تنازل', 'جزائر', 'ناس', 'سونتر', 'بروفايل', 'صوم']),
        ('متشرفين نورمال تيميمون حلقات تركعوا عدوانية يفيقولو وعليكم بصيرته بصيرتها عملها عملهم', 
         ['متشرف', 'نورم', 'تيميم', 'حلق', 'تركع', 'عدواني', 'يفيقول', 'وعل', 'بصيرت', 'بصيرت', 'عمل', 'عمل']),
        ('رايحا طحتو توحشتك تبقاو ستوري راهي رميته الزنزانة وجيبوتي', 
         ['رايح', 'طحت', 'توحشت', 'تبقا', 'ستور', 'راه', 'رميت', 'زنزان', 'وجيبوت']),
    ]

    for test in tests:
        print('-----------------------------------')
        print('tweet ', test[0])
        print('your norm ', normalize_text(test[0]).split())
        print('must be', test[1])


def _tweet_sim_test():
    tweets = get_tweets() # If it cannot find the file, pass its URL as argument

    tests = [
        (1, 2, 0.45652173913043487),
        (4, 120, 0.40744680851063825),
        (5, 10, 0.3381987577639752),
        (204, 211, 0.4728021978021977),
        (15, 30, 0.48148148148148145),
        (50, 58, 0.3531746031746032),
        (100, 300, 0.5277777777777778),
    ]

    for test in tests:
        print('-----------------------------------')
        print('tweet 1', tweets[test[0]])
        print('tweet 2', tweets[test[1]])
        print('your sim ', tweet_sim(tweets[test[0]], tweets[test[1]]))
        print('must be  ', test[2])




# TODO activate one test at the time
if __name__ == '__main__':
    # _word_sim_test()
    # _normalize_text_test()
    _tweet_sim_test()