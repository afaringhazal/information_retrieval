import heapq
import pickle
from typing import Dict, List
from preprocess import Document


id_to_posting_dict: Dict[str, List[Document]] = dict()
champion_id_to_postings: Dict[str, List[Document]] = dict()

id_to_posting_dict = pickle.load(open("../phase1/inverted_index", "rb"))

K = 1000


def base_doc_id(val:Document):
    return val.doc_id


def sort_by_doc_id(postings: List[Document]):
    return postings.sort(key=base_doc_id)


def champion():
    for word, posting_list in id_to_posting_dict.items():
        heapq.heapify(posting_list) # pop from maxheap
        if K <= len(posting_list):
            new_list_of_postings = [heapq.heappop(posting_list) for i in range(K)]
        else:
            new_list_of_postings = [heapq.heappop(posting_list) for i in range(len(posting_list))]
        sort_by_doc_id(new_list_of_postings)
        champion_id_to_postings[word] = new_list_of_postings
    print("finish champion_")
    with open('champion_', 'wb') as fpp:
        pickle.dump(champion_id_to_postings, fpp)


champion()
print("successful")
