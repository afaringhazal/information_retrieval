import pickle
from preprocess import Information, Document , Term
from typing import Dict, List, Set


id_to_posting_dict: Dict[str, List[Document]] = dict()


def convert_to_dictionary(terms: List[Term]):
    for term in terms:
        word = term.word
        doc_id = term.doc_id
        pos = term.position

        posting_dict = id_to_posting_dict.get(word)
        if posting_dict is None:
            id_to_posting_dict[word] = [Document(doc_id, [pos])]
        else:
            last_document = posting_dict[-1]
            if last_document is not None:
                if last_document.doc_id == doc_id:
                    last_document.add_position(pos)
                else:
                    posting_dict.append(Document(doc_id, [pos]))


terms_ = pickle.load(open("sort_file", "rb"))
convert_to_dictionary(terms_)
print("finish convert")
with open('inverted_index', 'wb') as fpp:
    pickle.dump(id_to_posting_dict, fpp)
