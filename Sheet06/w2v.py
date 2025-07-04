import time

from annoy.annoylib import Annoy
from sympy import closest_points

from voc import Vocabulary
from torch.utils.data import Dataset, DataLoader
import wandb
import json
import numpy as np
import argparse
import torch
import random
import torch.nn as nn
import pickle
import os
import annoy


# setup wandb logging


''' Dataset Parameters '''
#### small dataset
# DOCSPATH = './data/recipes-1000.json' # small dataset?
# MIN_OCCURRENCES = 30
# NDOCS = 1000
#### large dataset
DOCSPATH = './data/recipes.json'
MIN_OCCURRENCES = 80
NDOCS = 300000
WINDOWSIZE = 2

''' Model Parameters '''
MODELPATH = './model.pt'
D = 300                    # embedding dimension
BATCHSIZE = 2000
EPOCHS = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class W2VNet(nn.Module):

    def __init__(self, D, voc):
        super(W2VNet, self).__init__()

        self.U = torch.nn.Embedding(len(voc), D).to(device)
        self.V = torch.nn.Embedding(len(voc), D).to(device)

        nn.init.uniform_(self.U.weight, -1. / D, 1. / D)
        nn.init.uniform_(self.V.weight, -1. / D, 1. / D)
        
                    
    def forward(self, X):
        '''
        given a batch of token-ID pairs of shape (N,2),
        this should return the probabilies P(t|t') 
        for each pair (t,t'). 
        '''

        # maybe think about it
        u = self.U(X[:, 0]).to(device)
        v = self.V(X[:, 1]).to(device)

        u2 = torch.unsqueeze(u, 1)
        v2 = torch.unsqueeze(v, 2)

        dot_products = torch.bmm(u2, v2).flatten()
        probabilities = torch.sigmoid(dot_products)

        return probabilities

    def forward_doc(self, doc):
        '''
        use for the last exercise, to get an embedding
        for a whole document/recipe.
        '''

        doc = torch.tensor(doc).to(device)

        u_embedding = self.U(doc).to(device)
        v_embedding = self.V(doc).to(device)

        avg_emb = torch.cat((u_embedding, v_embedding), 0)

        return torch.mean(avg_emb, dim=0)


    
class W2VDataset(Dataset):
    
    def __init__(self, path_docs, voc, ndocs=None, npos=50, K=3, W=3):
        '''
        !!! NO NEED TO CHANGE !!!

        param path_docs: path to the documents JSON.
        voc:             the vocabulary (used to map tokens to integer IDs)
        ndocs:           if specified, use only the first 'ndocs' documents
        npos:            how many positive token pairs to sample per document
        K:               now many negative token pairs to sample per positive token pair
        W:               the window size, determines when two tokens are considered 'close'
                         (when sampling positive token pairs)
        
        '''

        self.npos = npos
        self.K    = K
        self.W    = W
        self.voc  = voc
        
        # load data
        docs = self._read_data(path_docs)
        print('read %d docs...' %(len(docs)))

        random.shuffle(docs)
        
        # apply vocabulary
        print('mapping tokens to ints...')
        if ndocs is None:
            ndocs = len(docs)
        self.docs = [None] * ndocs
        for i,doc in enumerate(docs[:ndocs]):
            self.docs[i] = self.voc.apply(doc)

        print('avg document length =', np.mean([len(d) for d in self.docs]))

        
    def _read_data(self, path):
        with open(path) as stream:
            docs_all = json.load(stream)
            docs = [d['instructions'] for d in docs_all]
            return docs

        
    def __getitem__(self, index):
        ''' 
        draws token pairs from Document no. 'index'.
        '''
        npos = self.npos
        K    = self.K
        W    = self.W

        doc = self.docs[index]

        X = torch.zeros((npos*(1+K),2), dtype=torch.long)
        y = torch.zeros(npos*(1+K))

        i = 0
        while i < npos*(1+K):

            # sample an anchor token
            anchor = random.randint(0,len(doc)-1)
            if doc[anchor] == Vocabulary.UNKNOWN:
                continue

            # sample a positive partner token
            rleft = range( max(0,anchor-W), anchor)
            rright = range( min(anchor+1,len(doc)-1), min(anchor+W+1,len(doc)) )
            posrange = list(rleft) + list(rright)
            xpos = random.choice( posrange )
            X[i:i+K+1,0] = doc[anchor]
            X[i,1] = doc[xpos]

            # sample K negative partner tokens
            for k in range(K):
                X[i+1+k,1] = random.randint(0,len(self.voc)-1)
            i += K+1

        y[0::1+K] = 1
        
        return X,y
    
    def __len__(self):
        ''''''
        return len(self.docs)

    def get_labels(self):
        return [self.voc.id2token[i] for i in range(len(self.voc))]




def train_and_save_vocabulary():
    with open(DOCSPATH) as stream:
        docs_all = json.load(stream)
        docs = [d['instructions'] for d in docs_all]

    voc = Vocabulary(min_occurrences=MIN_OCCURRENCES)
    voc.train(docs)
    print('Trained a vocabulary with %d tokens' %(len(voc)))
    for i in range(0,len(voc),100):
        print(i, voc.id2token[i])
    voc.save('voc_recipes.pkl')
    return voc




def run_training(voc, dataset, trial=None):

    learning_rate = 0.001

    wandb.config.learning_rate = learning_rate

    dataloader = DataLoader(dataset,
                            batch_size=BATCHSIZE,
                            shuffle=True,
                            num_workers=8)

    net = W2VNet(D=D, voc=voc).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    bce_loss = torch.nn.BCELoss()

    for epoch in range(EPOCHS):
        start_time = time.time()
        # store a checkpoint of the model
        torch.save(net.state_dict(), MODELPATH)

        for (i, batch) in enumerate(dataloader):
            X, y = batch
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            prob = net.forward(X.view(-1, 2))
            loss = bce_loss(prob, y.flatten())
            loss.backward()
            optimizer.step()

            wandb.log({'loss': loss.item(),
                       'epoch': epoch+1,})

        end_time = time.time()
        print('Time for one epoch: ', end_time-start_time)
        print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item()}')

    emb = net.U.weight.cpu().detach().numpy()

    pickle_data = (emb, voc)

    with open (f'results/result_epoch_{epoch+1}.pkl', 'wb') as f:
        pickle.dump(pickle_data, f)
    print('Finished Training')



    
if __name__ == "__main__":

    wandb.init(project='w2v')
    wandb.config.epochs = EPOCHS
    wandb.config.batch_size = BATCHSIZE

    # parse command line arguments (no need to touch)
    parser = argparse.ArgumentParser(description='word2vec on chefkoch.de recipes.')
    parser.add_argument('--train', help="train the w2v model", action='store_true')
    parser.add_argument('--apply', help="apply the w2v model", action='store_true')

    args = parser.parse_args()
        
    if args.train:

        print('building vocabulary...')
        voc = train_and_save_vocabulary()
        print('done.')

        # initialize dataset
        dataset = W2VDataset(DOCSPATH, voc, ndocs=NDOCS, K=10, W=WINDOWSIZE)

        print('training...')
        run_training(voc, dataset)

    if args.apply:

        with open(DOCSPATH) as stream:
            docs_all = json.load(stream)
            docs = [d['instructions'] for d in docs_all]

        print('building vocabulary...')
        voc = train_and_save_vocabulary()
        print('done.')

        net = W2VNet(D=D, voc=voc).to(device)

        net.load_state_dict(torch.load(MODELPATH))

        embs = []
        for doc in docs:
            tokenize = np.array(voc.apply(doc))
            tokenize = tokenize[tokenize != 0]

            emb = net.forward_doc(tokenize).cpu().detach().numpy()
            embs.append(emb)

        embs = np.array(embs)
        annoy_index = annoy.AnnoyIndex(len(embs[0]), 'euclidean')

        for i, emb in enumerate(embs):
            annoy_index.add_item(i, emb)

        annoy_index.build(10)
        print(docs_all[500]['name'])

        for i, emb in enumerate(embs):
            closest_indices = annoy_index.get_nns_by_vector(emb, 5)
            print(f"Recipe {docs_all[i]['name']} is similar to recipes {', '.join(map(str, [{docs_all[index]['name']} for index in closest_indices]))}")