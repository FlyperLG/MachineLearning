import pickle
import sys
import numpy as np
import pandas as pd
import torch
from umap import UMAP
import plotly.express as px

''' 
     !!! NO NEED TO CHANGE (I THINK...) !!!
'''

if len(sys.argv) < 1:
    print('USAGE: vis.py <EMBEDDINGS_AND_LABELS.PKL>')

with open(sys.argv[1], 'rb') as stream:
    emb,voc = pickle.load(stream)

emb = torch.tensor(emb)
N,D = emb.shape
emb = emb.detach().numpy()
labels = [voc.id2token[i] for i in range(N)]

# reduce to 3D with dimensionality reduction
umap = UMAP(n_components=3, random_state=42)
emb3d = umap.fit_transform(emb)

# create a dataframe for plotly
df = pd.DataFrame(emb3d, columns=['x1', 'x2', 'x3'])
df['label'] = labels
df['hover_label'] = labels

# Plotly 3D scatter plot with hover labels
fig = px.scatter_3d(
    df, 
    x='x1', 
    y='x2', 
    z='x3',
    size=None,
    hover_name='label',
    title="Word2vec Embeddings"
)
fig.update_traces(marker=dict(size=2))

fig.show()
