import json
import pickle
from phase1.preprocess import Term, Document
from typing import Dict, List, Set

data = {}
dict_to_idf: Dict[str, List[Document]] = dict()


def retrieve_doc():
    file = open('../phase1/IR_data_news_12k.json')
    data_file = json.load(file)
    file.close()
    for doc_Id, body in data_file.items():
        data[doc_Id] = {}
        data[doc_Id]['title'] = body['title']
        data[doc_Id]['content'] = body['content']
        data[doc_Id]['url'] = body['url']


retrieve_doc()
All_documents = len(data)
print("find all documents")
with open('number_of_all_documents', 'wb') as fpp:
    pickle.dump(All_documents, fpp)
# we should add idf for any words of query by use : log(All_document / size_of postings)