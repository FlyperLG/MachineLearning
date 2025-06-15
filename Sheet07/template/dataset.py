from cProfile import label

import torch
import numpy as np
from torch.utils.data import Dataset
import sys
import json

from transformers import AutoTokenizer

import config as CONFIG




class RecipERTDataset(Dataset):

    def __init__(self, class2int, recipes):
        super(RecipERTDataset, self).__init__()
        self.class2int = class2int
        self.recipes = recipes
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG.MODEL_ID)
        # FIXME: more attribuets?

    def __getitem__(self, index):
        '''
            returns a recipe from the dataset, as a dictionary of the following form:

            {
              'tokens': tokens [0, 17, 43, 563, 372, ...]
              'labels': ...
            }

            where
            - tokens is a sequence of token IDs produced by the BERT tokenizer, and
            - labels is a boolean torch tensor with as many entries as there are recipe classes
                     (each entry indicates if the recipe belongs to a certain category).
        '''
        recipe = self.recipes[index]
        #print(recipe)
        tokens = self.tokenizer(recipe['instructions'])
        input_ids = tokens['input_ids'][:CONFIG.MAX_LENGTH] # ToDo: Last Token check

        categories = recipe['recipeCategory']
        labels = [key in categories for key in self.class2int.keys()]

        #print(labels)

        return {'tokens': torch.tensor(input_ids), 'labels': torch.tensor(labels)}

    def __len__(self):
        return len(self.recipes)



def train_vocabulary_of_recipe_classes(recipes, nclasses):
    '''
        given a collection of recipes, identify the most frequent recipe classes.
        return a dictionary mapping each class to an integer ID, of the following form:

        class2int = {
          'Hauptspeise': 0,
          'Überbacken': 1,
          ...
        }
    '''
    categories = np.concatenate([recipe['recipeCategory'] for recipe in recipes])

    # Count occurrences of each unique category
    unique_categories, counts = np.unique(categories, return_counts=True)

    sorted_indices = np.argsort(-counts)[:nclasses]
    top_categories = unique_categories[sorted_indices]
    class2int = dict(zip(top_categories, np.arange(0, nclasses, 1, dtype=int)))

    return class2int


if __name__ == '__main__':
    print(sys.argv[1])
    with open(sys.argv[1], 'r') as f:
        recipes = json.load(f)
    class2int = train_vocabulary_of_recipe_classes(recipes, CONFIG.NCLASSES)
    dataset = RecipERTDataset(class2int, recipes)
    print("============================================")
    print( dataset[0] )
    print("============================================")
    print( dataset[1] )
