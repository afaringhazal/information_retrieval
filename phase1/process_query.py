import pickle
from typing import Dict, List, Set
import json
import logging
from phase1.preprocess import Document, tokenize, stem_v2, normalize, stop_word_v2

id_to_posting_dict: Dict[str, List[Document]] = dict()
id_to_posting_dict = pickle.load(open("inverted_index", "rb"))

# phase 3


class Answer:
    # doc_id, positions: list,
    def __init__(self, doc_id, positions: List[int], rank=1):
        self.doc_id = doc_id
        self.positions = positions
        self.rank = rank

    def add_position(self, position):
        self.positions.append(position)

    def add_rank(self):
        self.rank += 1


doc_id_to_title_and_url = {}


def complete_result():
    """
    in this method , make a structure of result that have doc id to title and url
    :return: none
    """
    file = open('IR_data_news_12k.json')
    data_file = json.load(file)
    file.close()
    for doc_Id, body in data_file.items():
        doc_id_to_title_and_url[doc_Id] = {}
        doc_id_to_title_and_url[doc_Id]['title'] = body['title']
        doc_id_to_title_and_url[doc_Id]['url'] = body['url']
        doc_id_to_title_and_url[doc_Id]['content'] = body['content']


def is_not(s):
    """
    find not action in query
    :param s: the word that we want to be not in search
    :return: true or false
    """
    if s == '!':
        return True
    return False


class Not:
    """
    This class is to introduce the Not feature
    """
    pass


def is_concat(s):
    """
    find the concat action in query
    :param s: the string that we want to concat another string
    :return: true or false
    """
    if s == '"':
        return True
    return False


class Concat:
    """
    this class is to introduce Concat feature
    """
    pass


class Query:
    def __init__(self, word: str, actions: list):
        self.word = word
        self.actions = actions

    def set_word(self, word):
        self.word = word

    def add_action(self, action):
        self.actions.append(action)


def process_query(query):
    """
    this method separate word by tokenizer and find their actions(NOT and CONCAT)
    :param query: String of word by some notation like ! that shows action of word
    :return: the list of Query( that contain word and actions)
    """
    query = normalize(query)
    new_tokens_list = tokenize(query)
    tokens_list = []
    for token in new_tokens_list:
        token = stem_v2(token)
        if not stop_word_v2(token):
            if not token == '':
                tokens_list.append(token)
    new_query_list = []
    ac = []
    word: str = ""
    check_end = False
    correct_word = True
    for j in range(len(tokens_list)):
        if is_not(tokens_list[j]):
            ac.append(Not())
            continue
        if is_concat(tokens_list[j]):
            if not check_end:
                check_end = True
                ac.append(Concat())
                continue
            else:
                check_end = False
        if check_end:
            word = word + tokens_list[j] + " "
            correct_word = False
            continue
        if correct_word:
            word = tokens_list[j]
        new_query_list.append(Query(word, actions=ac))
        word = ""
        correct_word = True
        ac = []

    return new_query_list


def convert_document_to_answer(postings):
    answer_list: List[Answer] = list()
    for posting in postings:
        answer_list.append(Answer(posting.doc_id, posting.positions))
    return answer_list


def my_actions(val: Query):
    return len(val.actions) != 0


def sort_query_list_by_action(query_list: List[Query]):
    return query_list.sort(key=my_actions)


def rank_func(val: Answer):
    return val.rank


def sort_total_answer_by_rank(total_answer_by_differ_rank: List[Answer]):
    total_answer_by_differ_rank.sort(key=rank_func, reverse=True)


def create_answer(query_list: List[Query]):
    first = 1
    i = 0
    total_answer: List[Answer] = list()
    total_answer_by_differ_rank: Set[Answer] = set()
    while i < len(query_list):
        action_total = query_list[i].actions
        if len(action_total) == 0:
            posting = id_to_posting_dict.get(query_list[i].word)
            convert_doc_to_ans = convert_document_to_answer(posting)
            total_answer = intersect_and(convert_doc_to_ans, list(total_answer), first)
            total_answer_by_differ_rank.update(total_answer)
            first = 0
        else:
            for action_1 in action_total:
                if isinstance(action_1, Not):
                    posting = id_to_posting_dict.get(query_list[i].word)
                    convert_doc_to_ans = convert_document_to_answer(posting)
                    total_answer = intersect_not(convert_doc_to_ans, list(total_answer))
                    total_answer_by_differ_rank.update(total_answer)
                elif isinstance(action_1, Concat):
                    concat_words = query_list[i].word
                    concat_words = normalize(concat_words)
                    new_concat_list = tokenize(concat_words)
                    concat_list = []
                    for concat_word in new_concat_list:
                        concat_word = stem_v2(concat_word)
                        if not stop_word_v2(concat_word):
                            if not concat_word == '':
                                concat_list.append(concat_word)
                    concat_l = 0
                    len_first_and_second_word = 0
                    first_concat = 1
                    while concat_l < len(concat_list):
                        new_posting = id_to_posting_dict.get(concat_list[concat_l])
                        new_answer = convert_document_to_answer(new_posting)
                        total_answer, len_first_and_second_word = intersect_concat(total_answer, new_answer,
                                                                                   len_first_and_second_word,
                                                                                   len(concat_list[concat_l]),
                                                                                   first_concat=first_concat)
                        concat_l += 1
                        first_concat = 0
                        total_answer_by_differ_rank.update(total_answer)
                else:
                    logging.error("The action under consideration is not already defined")
        i += 1
    return total_answer_by_differ_rank


def intersect_concat(first_postings: List[Answer], second_postings: List[Answer], len_first_word: int,
                     len_second_word: int, first_concat):
    answers: List[Answer] = list()
    i = j = 0
    if len_first_word == 0 and first_concat == 1:
        return intersect_and(first_postings=second_postings, second_postings=first_postings, first=1), len_second_word
    while i < len(first_postings) and j < len(second_postings):
        if first_postings[i].doc_id == second_postings[j].doc_id:
            document_id = first_postings[i].doc_id
            positions_1: List[int] = first_postings[i].positions
            positions_2: List[int] = second_postings[j].positions
            i_pos_1 = j_pos_2 = 0
            while i_pos_1 < len(positions_1) and j_pos_2 < len(positions_2):
                if positions_1[i_pos_1] < positions_2[j_pos_2]:
                    if positions_1[i_pos_1] + len_first_word == positions_2[j_pos_2]:
                        for answer in answers:
                            if answer.doc_id == document_id:
                                answer.add_position(positions_1[i_pos_1])
                                break
                        else:
                            new_rank = first_postings[i].rank + second_postings[j].rank
                            answers.append(Answer(doc_id=document_id, positions=[positions_1[i_pos_1]], rank=new_rank))
                        i_pos_1 += 1
                        j_pos_2 += 1
                    else:
                        i_pos_1 += 1
                elif positions_1[i_pos_1] == positions_2[j_pos_2]:
                    logging.error(f'Two different words have the same positions => doc id : {document_id} , '
                                  f'pos1 :{positions_1[i_pos_1]} , pos2 : {positions_2[j_pos_2]}')
                else:
                    j_pos_2 += 1
            i += 1
            j += 1
        elif first_postings[i].doc_id < second_postings[j].doc_id:
            i += 1
        else:
            j += 1
    return answers, len_first_word + len_second_word


def intersect_and(first_postings: List[Answer], second_postings: List[Answer], first):
    i_ = 0
    j_ = 0
    answers: List[Answer] = list()
    if len(second_postings) == 0 and first:
        answers.extend(first_postings)
        return answers
    while i_ < len(first_postings) and j_ < len(second_postings):
        if first_postings[i_].doc_id == second_postings[j_].doc_id:
            document_id = first_postings[i_].doc_id
            positions_1: List[int] = first_postings[i_].positions
            positions_2: List[int] = second_postings[j_].positions
            merge_pos: List[int] = positions_1 + positions_2
            merge_pos.sort()
            new_pos_include_same_value = merge_pos
            new_pos = list(set(new_pos_include_same_value))
            new_rank = first_postings[i_].rank + second_postings[j_].rank + 1
            answers.append(Answer(doc_id=document_id, positions=new_pos, rank=new_rank))
            i_ += 1
            j_ += 1
        elif first_postings[i_].doc_id < second_postings[j_].doc_id:
            i_ += 1
        else:
            j_ += 1
    return answers


def intersect_not(not_in_postings: List[Answer], in_postings: List[Answer]):
    i_ = 0
    j_ = 0
    answers: List[Answer] = list()
    while i_ < len(not_in_postings) and j_ < len(in_postings):
        if not_in_postings[i_].doc_id == in_postings[j_].doc_id:
            i_ += 1
            j_ += 1
        elif not_in_postings[i_].doc_id < in_postings[j_].doc_id:
            i_ += 1
        else:
            in_postings[j_].add_rank()
            answers.append(in_postings[j_])
            j_ += 1

    for end_list in range(j_, len(in_postings)):
        in_postings[end_list].add_rank()
        answers.append(in_postings[end_list])
    return answers


val_user = input("Enter your value : ")
query_list_user = process_query(val_user)
sort_query_list_by_action(query_list_user)
answer_user = create_answer(query_list=query_list_user)
list_answer_usr = list(answer_user)
sort_total_answer_by_rank(list_answer_usr)
complete_result()
max_ = 5
for ans in list_answer_usr:
    if max_ == 0:
        break
    print(doc_id_to_title_and_url[str(ans.doc_id)]['title'])
    print(doc_id_to_title_and_url[str(ans.doc_id)]['url'])
    print(doc_id_to_title_and_url[str(ans.doc_id)]['content'])
    print(ans.doc_id)
    print(ans.positions)
    print(ans.rank)
    print("-----------------")
    max_ -= 1
