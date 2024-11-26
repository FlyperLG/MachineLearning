import os
import json
import pickle
import argparse
from collections import Counter

VOC_PATH = '_voc.pkl'
MIN_OCCURRENCES = 80

class Vocabulary:

    ''' 
        !!! NO NEED TO CHANGE !!!

    turns an input string into a sequence of token indices:
    
           the quick brown fox

           >  1  8  56   74   

        the vocabulary is 'learned' (i.e., the set of tokens
        is built) on a corpus of training documents.
    '''

    # the index reserved for unknown tokens
    UNKNOWN = 0

    def __init__(self, min_occurrences=2):
        # maps tokens ('brown') to integers (56)
        self.token2id = {}
        # maps integers (56) to tokens ('brown')
        self.id2token = {}
        # minimum number of occurrences in corpus
        # required for a token to be part of the vocabulary.
        self.min_occurrences = min_occurrences
        
    def __len__(self):
        return len(self.id2token)

    def train(self, docs):

        docs = [self._tokenize(doc) for doc in docs]

        # count how frequent each token is
        counts = Counter()
        for doc in docs:
            counts.update(doc)

        # filter infrequent tokens
        counts = {t:count for t,count in counts.items()
                  if count >= self.min_occurrences}

        # sort from frequent to infrequent
        counts = sorted(counts.items(), key=lambda x:x[1], reverse=True)

        # build index
        self.token2id = {}
        self.id2token = {}
        for i,(token,count) in enumerate(counts):
            self.token2id[token] = i+1
            self.id2token[i+1] = token

        # map unknown tokens to '***'
        self.id2token[Vocabulary.UNKNOWN] = '***'

        
    def _tokenize(self, doc):
        # preprocessing
        doc = doc.lower()
        doc = doc.replace('-', ' - ')
        doc = doc.replace('"', '')
        doc = doc.replace('(', '')
        doc = doc.replace(')', '')
        doc = doc.replace('!', ' ')
        doc = doc.replace('.', ' ')
        doc = doc.replace(';', ' ')
        doc = doc.replace(':', ' ')
        doc = doc.replace(',', ' ')
        doc = doc.split()
        return doc

    
    def apply(self, doc):
        tokens = self._tokenize(doc)
        return [self.token2id.get(t, Vocabulary.UNKNOWN) for t in tokens]

    
    def unapply(self, ids):
        return ' '.join([self.id2token[i] for i in ids])

    
    def save(self, path):
        with open(path, 'wb') as stream:
            voc = self.min_occurrences, self.token2id, self.id2token
            pickle.dump(voc, stream)

    def load(self, path):
        try:
            with open(path, 'rb') as stream:
                voc = pickle.load(stream)
                self.min_occurrences, self.token2id, self.id2token = voc
        except:
            print('Could not load vocabulary from', path)



def tiny_test():
    
    voc = Vocabulary(min_occurrences=2)

    docs = [
        'Ich bin doof.',
        'ich bin gut'
    ]
    
    voc.train(docs)
    voc.save('_voc.pkl')

    print(voc.token2id)
    print(voc.id2token)

    x = voc.apply('ich bin super')
    print(x)
    print(voc.unapply(x))

    # test loading and storing
    voc2 = Vocabulary()
    voc2.load('_voc.pkl')
    print(voc2.id2token)
    
    x = voc2.apply('ich bin super')
    print(x)
    print(voc2.unapply(x))



if __name__ == "__main__":

    # parse command line arguments (no need to touch)
    parser = argparse.ArgumentParser(description='a vocabulary for ML on documents.')
    parser.add_argument('--train', help="train the vocabulary", action='store_true')
    parser.add_argument('--apply', help="apply the vocabulary", action='store_true')

    args = parser.parse_args()

    with open('recipes.json') as stream:
        docs_all = json.load(stream)
        docs = [d['instructions'] for d in docs_all]
    
    if args.train:

        voc = Vocabulary(min_occurrences=MIN_OCCURRENCES)
        voc.train(docs)
        print('Trained a vocabulary with %d tokens.' %(len(voc)))
        for i in range(0,len(voc),100):
            print(i, voc.id2token[i])
        voc.save(VOC_PATH)


    if args.apply:
        
        voc = Vocabulary(min_occurrences=MIN_OCCURRENCES)
        voc.load(VOC_PATH)

        for doc in [docs[17]]:

            print('--------------------')
            print('ORIGINAL DOCUMENT')
            print('--------------------')
            print(doc)
            print()
            
            print('--------------------')
            print('DOCUMENT MAPPED TO TOKENS')
            print('--------------------')
            x = voc.apply(doc)
            print(x)
            print()

            print('--------------------')
            print('DOCUMENT RECONSTRUCTED (MAPPED TO VOCABULARY AND BACK)')
            print('--------------------')
            print(voc.unapply(x))
            print()
