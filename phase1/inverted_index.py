import pickle
from typing import List
from preprocess import Term


def my_function(val: Term):
    """
    list.sort() has a key parameter to specify a function (or other callable)
    to be called on each list element prior to making comparisons
    :param val: term
    :return: word of the term
    """
    return val.word


def sort_by_term(terms: List[Term]):
    """
    in this method sort by term and doc_id because a sorting algorithm is stable
    :param terms: a list of term that sorted by doc_id
    :return: a list of term that sorted by term and doc_id
    """
    terms.sort(key=my_function)
    print("finish sort")
    with open('sort_file', 'wb') as fpp:
        pickle.dump(terms, fpp)


term = pickle.load(open("listfile", "rb"))
sort_by_term(term)
