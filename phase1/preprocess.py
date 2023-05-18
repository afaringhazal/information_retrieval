import json
import math

from parsivar import Tokenizer, Normalizer, FindStems
from hazm import stopwords_list, Stemmer
from typing import List, Dict
import pickle
import string

stop_words = stopwords_list()
punctuation = ['.','/',',','[',']',':','"','<','>','\\','!','@','#','$','%','^','&','*','(',')','-','+','=','\'','_','،' ,'!!' ,
               '!!','!>','!»', '?!','!؟','".','",','")','"(','%)',')-',').','):','):-','),',')،','»']


def tokenize(doc_string):
    """
    this method use tokenizer from parsivar and
    cut character sequence into word tokens.
    :param doc_string: text
    :return: a list of tokens
    """
    my_tokenizer = Tokenizer()
    return my_tokenizer.tokenize_words(doc_string=doc_string)


def normalize(doc_string):
    """
    this method use Normalizer from parsivar and
    map text and query term to same form
    :param doc_string: text
    :return: correct form of the text
    """
    my_normalizer = Normalizer()
    return my_normalizer.normalize(doc_string=doc_string)


def stem(word):
    """
    this method use FindSteam form parsivar and
    mach different forms of a root
    :param word: word
    :return: the root of the word
    """
    my_stemmer = FindStems()
    return my_stemmer.convert_to_stem(word)


def stem_v2(word):
    stemmer = Stemmer()
    return stemmer.stem(word)


def stop_word(words):
    """
    in this method we omit very common words
    :param words: a list of words
    :return: a list of words without common words
    """
    stop_words_list = ['حتی', 'حتما', 'خصوصا', 'اگر', 'زیرا', 'اکنون', 'البته', 'را', 'با', 'برای', 'یا', 'این', 'این',
                       'از', 'به', 'در', 'اما', 'انتهای', '/', 'پیام', '.', ':', '،', 'و']
    new_list = []
    for word in words:
        if word in stop_words_list:
            pass  # do nothing
        else:
            new_list.append(word)
    return new_list


def stop_word_v2(word):
    if word in stop_words:
        return True
    return False


#  phase1


def preprocessing_phase1():
    file = open('IR_data_news_12k.json')
    data = json.load(file)
    # test normalization
    value = data['1']['content']
    print(value)
    print("after normalization : ")
    normalization_content = normalize(value)
    print(normalization_content)

    # test tokenizer
    print("after tokenizer : ")
    token_list = tokenize(normalization_content)
    print(token_list)

    # test stemming
    print("after steaming : ")
    root_list = []
    for word in token_list:
        root = stem(word)
        root_list.append(root)

    print(root_list)

    # test stop word
    print("after stop words:")
    stop_words_ = stop_word(root_list)
    print(stop_words_)

    file.close()


#  phase 2  inverted index : in this part we must do three parts
#  ( 1- token sequence 2-sort by term (and doc_id) 3- convert to dictionary

class Term:
    def __init__(self, doc_id, word, position):
        self.word = word
        self.doc_id = doc_id
        self.position = position


class Document:
    def __init__(self, doc_id: int, positions: List[int]):
        self.doc_id = doc_id
        self.positions = positions

    def add_position(self, position):
        self.positions.append(position)

    @property
    def pos_fre(self):
        return len(self.positions)

    @property
    def tf(self):
        return 1+math.log(len(self.positions))

    def __lt__(self, other):
        return self.tf > other.tf


class Information:
    def __init__(self, letter, frequency=1):
        self.letter = letter
        self.frequency = frequency

    def add_frequency(self):
        self.frequency += 1

    def __eq__(self, other):
        if other is not None:
            return self.letter == other.letter


# token sequence

# word_to_doc_id_to_position

def preprocessing_phase2():
    """
    firs part of inverted index is token sequence and in this part
    do preprocessing on content
    the preprocessing is normalization , tokenizer , stemming , stop word
    :return: a sequence of word( that sort by doc_id)
    """
    file = open('IR_data_news_12k.json')
    data_file = json.load(file)
    file.close()
    data = {}
    for doc_Id, body in data_file.items():
        data[doc_Id] = {}
        data[doc_Id]['title'] = body['title']
        data[doc_Id]['content'] = body['content']
        data[doc_Id]['url'] = body['url']

    number_of_doc = 0
    terms = []
    for doc_Id in data.keys():
        value = data[doc_Id]['content']
        # preprocessing content : normalization , tokenizer , stemming , strop word
        normal_text = normalize(value)
        # tokenizer (create term)
        token_list = tokenize(normal_text)
        number_of_position = 0
        for token in token_list:
            term = Term(number_of_doc, token, number_of_position)
            if stop_word_v2(term.word):
                number_of_position += len(token)
                continue
            term.word = stem_v2(term.word)
            if not term.word == '':
                if term.word not in string.punctuation:
                    if term.word not in punctuation:
                        terms.append(term)
            number_of_position += len(token)
        number_of_doc += 1
    print("finish tokenizer and normalizer ,stemming , del stop word")
    print(f'size of term {len(terms)}')
    with open('listfile', 'wb') as fpp:
        pickle.dump(terms, fpp)
    return terms


# preprocessing_phase2()