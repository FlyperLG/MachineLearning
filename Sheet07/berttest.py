import torch
from transformers import AutoModel, PreTrainedModel, AutoTokenizer

MODEL_ID = 'bert-base-german-cased'

transformer = AutoModel.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

s = 'Mein Name ist Bond. James Bond.'

tokenization_result = tokenizer(s)

tokens = tokenization_result['input_ids']
print(tokens)


tokens2words = tokenization_result.words()
print(tokens2words)

tokenizer.decode(tokens)

print('We input a token sequence of length:', len(tokens))
print(tokens)

tokens = torch.tensor([tokens])
print(tokens.shape)
output = transformer(tokens)
y = output.last_hidden_state
print(y.shape)

# three sentences of different length
s1 = 'Der Rang einer Matrix entspricht der Anzahl ihrer nicht-nullen Eigenwerte.'
s2 = 'Gegeben eine positiv-semidefinite Matrix vom Rang n, kann gezeigt werden, dass sie invertierbar ist.'
s3 = 'Wenn man über militärischen Rang spricht, müsste man General, Major und Sergeant einschließen.'


# we tokenize them, obtaining token sequences of different length
tokens = tokenizer([s1,s2,s3], padding=True)['input_ids']

print('first sentence:', len(tokens[0]), 'tokens.')
print('second sentence:', len(tokens[1]), 'tokens.')
print('third sentence:', len(tokens[2]), 'tokens.')

tokens = torch.tensor(tokens)
output = transformer(tokens)

xbert = output.last_hidden_state

# rank in sentence 1/2/3
print([tokenizer.decode(t) for t in tokens[0]])
print([tokenizer.decode(t) for t in tokens[1]])
print([tokenizer.decode(t) for t in tokens[2]])
xbert1 = xbert[0,2,:]   # rank in sentence 1
xbert2 = xbert[1,11,:]  # rank in sentence 2
xbert3 = xbert[2,5,:]   # rank in sentence 3

# the jaguars in the first two sentences are indeed considered more similar by BERT.
def check_similarity(sx,sy,xbertx,xberty):
    print('>', sx)
    print('>', sy)
    print('>>> distance:', torch.sum((xbertx-xberty)**2))
    print()

check_similarity(s1,s2,xbert1,xbert2)
check_similarity(s1,s3,xbert1,xbert3)
check_similarity(s2,s3,xbert2,xbert3)

if __name__ == '__main__':
    print("Transformer")