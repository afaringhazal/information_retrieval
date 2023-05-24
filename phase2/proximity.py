import json
import logging
import math
import pickle
from typing import Dict, List
# import re
import nltk
# nltk.download('punkt')
from phase1.preprocess import Document
from phase1.process_query import process_query, Query
query_ = "باشگاه"
query_list_ = process_query(query=query_)
print(query_list_)



id_to_posting_dict: Dict[str, List[Document]] = dict()
champion_id_to_postings: Dict[str, List[Document]] = dict()

id_to_posting_dict = pickle.load(open("../phase1/inverted_index", "rb"))
champion_id_to_postings = pickle.load((open("champion_", "rb")))

Number_of_answer = 5
index_elimination = 0.75
answer_of_documents: Dict[Document, float] = dict()


doc_id_to_title_and_url_and_content = {}

def complete_result():
    """
    in this method , make a structure of result that have doc id to title and url
    :return: none
    """
    file = open('../phase1/IR_data_news_12k.json')
    data_file = json.load(file)
    file.close()
    for doc_Id, body in data_file.items():
        doc_id_to_title_and_url_and_content[doc_Id] = {}
        doc_id_to_title_and_url_and_content[doc_Id]['title'] = body['title']
        doc_id_to_title_and_url_and_content[doc_Id]['url'] = body['url']
        doc_id_to_title_and_url_and_content[doc_Id]['content'] = body['content']

complete_result()

def calculate_idf(tf_wt_query_dict):
    total_documents = pickle.load(open("number_of_all_documents", "rb"))
    for query in tf_wt_query_dict.keys():
        document_frequency = len(id_to_posting_dict[query.word])
        query.add_action(math.log(total_documents/document_frequency))  # calculate idf by use log(N/df)
    return tf_wt_query_dict

def normalize(tf_query_dict):
    total_length = 0
    for query, val in tf_query_dict.items():
        total_length += val * val

    for query, val in tf_query_dict.items():
        tf_query_dict[query] = val / math.sqrt(total_length)

    return tf_query_dict


def calculate_query_weight(query_list: List[Query]):
    tf_wt_query_dict = {i: query_list.count(i) for i in query_list}  # tf : term frequency
    new_tf_query_dict = calculate_idf(tf_wt_query_dict)
    for query, value in new_tf_query_dict.items():
        value = query.get_idf * value
        new_tf_query_dict[query] = value
    return normalize(new_tf_query_dict)


def sort_and_separate_query(query_dict):
    query_list = []
    value_list = []
    for q,v in query_dict.items():
        query_list.append(q)
        value_list.append(v)
    return query_list, value_list


def normalize_document(satisfy_word_dw: List[float]):
    total_value = 0

    for i in range(len(satisfy_word_dw)):
        total_value += satisfy_word_dw[i] * satisfy_word_dw[i]
    for i in range(len(satisfy_word_dw)):
        satisfy_word_dw[i] = satisfy_word_dw[i] / math.sqrt(total_value)

    return satisfy_word_dw


def cosine(satisfy_word_dw: List[float], final_query_wt: List[float]):
    satisfy_word_dw = normalize_document(satisfy_word_dw)
    score_document = 0
    for i in range(len(final_query_wt)):
        score_document += satisfy_word_dw[i] * final_query_wt[i]

    return score_document


def score_cosine(satisfy_word_dw: List[float], final_query_wt: List[float]):
    return cosine(satisfy_word_dw=satisfy_word_dw, final_query_wt=final_query_wt)


def score_jacart(document, number_of_intersect_document, final_query_wt):
    content = doc_id_to_title_and_url_and_content[str(document.doc_id)]['content']
    list_query = process_query(content)
    totals_word_in_document = len(list_query) + len(final_query_wt)

    return number_of_intersect_document / totals_word_in_document


def create_answer(query_list: List[Query], type_score, final_query_wt: List[float]):
    query_size = len(query_list)
    pointers = [0]*query_size
    documents_champion = []
    documents_total = []
    satisfy_word_dw = [0]*query_size
    for i in range(query_size):
        documents_champion.append(champion_id_to_postings.get(query_list[i].word))
        documents_total.append(id_to_posting_dict.get(query_list[i].word))

    for j in range(1000):
        break_while = 0
        number_of_satisfy_elimination = 0
        value = 0
        min = 0
        Jacart_sum_documents = 0
        # in this for we find correct min and number of value that equals by min
        for i in range(query_size):
            accessible_size = len(documents_champion[i])
            if accessible_size < pointers[i]:
                break_while += 1
                continue
            value = documents_champion[i][int(pointers[i])]
            if i == 0:
                min = value
                satisfy_word_dw[i] = value.tf
                number_of_satisfy_elimination += 1
                Jacart_sum_documents += len(value.positions)

            elif min.doc_id == value.doc_id:
                number_of_satisfy_elimination += 1
                satisfy_word_dw[i] = value.tf
                Jacart_sum_documents += len(value.positions)
            elif min.doc_id > value.doc_id:
                min = value
                satisfy_word_dw = [0]*query_size  # reset satisfy word
                satisfy_word_dw[i] = value.tf
                number_of_satisfy_elimination = 0
                Jacart_sum_documents = 0
        if number_of_satisfy_elimination/query_size >= index_elimination:
            if type_score == 0:
                value_of_document = score_cosine(satisfy_word_dw, final_query_wt) #calculate cosine or Jacart , document ,
            else:
                value_of_document = score_jacart(min, Jacart_sum_documents, final_query_wt)
            answer_of_documents[min] = value_of_document
        satisfy_word_dw = [0]*query_size # reset after work
        if break_while == len(query_list):
            break
        # now we move pointer forward
        for i in range(query_size):
            accessible_size = len(documents_champion[i])
            if accessible_size < pointers[i]:
                continue
            value = documents_champion[i][int(pointers[i])]
            if value.doc_id == min.doc_id:
                pointers[i] += 1
            elif value.doc_id < min.doc_id:
                logging.error("Minimum calculations are wrong")


def sort_answer(dic: Dict[Document, float]):
    return sorted(dic.items(), key=lambda x: x[1], reverse=True)







final_query_dict = calculate_query_weight(query_list_)
query_li, value_li =sort_and_separate_query(final_query_dict)


create_answer(query_list=query_li, type_score=0, final_query_wt=value_li)
answer_of_documents_ = sort_answer(answer_of_documents)
for document, score_ in answer_of_documents_:
    print(f'doc_id : {document.doc_id} , score : {score_}')
    print("len content  : ")
    print(len(process_query(doc_id_to_title_and_url_and_content[str(document.doc_id)]['content'])))
    print("title :")
    print(doc_id_to_title_and_url_and_content[str(document.doc_id)]['title'])
    content_ =doc_id_to_title_and_url_and_content[str(document.doc_id)]['content']
    # sentences = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', content_)
    sentences = nltk.sent_tokenize(content_)
    # context_sentences = []
    for sentence in sentences:
        if any(word in sentence for word in query_):
            print(sentence)
            print("********************")
            # context_sentences.append(sentence)
    # print(context_sentences)
    print("--------------------------------------")
    Number_of_answer -= 1
    if Number_of_answer == 0:
        break


print("successful!")
