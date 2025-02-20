from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
import config as CONFIG
from model import RecipERT
import sys
import os
import torch
import pickle
import json
from dataset import train_vocabulary_of_recipe_classes, RecipERTDataset


def collate_fn_pad(batch):
    '''
    !!! NO NEED TO CHANGE !!!

    this function collates multiple recipes/objects (as returned by
    a dataset's __getitem__() method) into a single batch.
    To do so, the token sequences are padded to the length of the longest
    recipe in the sequence.
    @param batch: a batch of inputs, each a dictionary containing tokens and labels.
    @type batch: list<dict>

    @returns: a batch, of the form

            {
                'tokens': tokens,
                'labels': labels,
                'attn_mask': attn_mask
            }

            where
            - tokens is a tensor of tokens (shape BATCHSIZE x SEQUENCE_LENGTH).
            - labels is a tensor of class labels (shape BATCHSIZE x #CLASSES).
            - attn_mask is a tensor of tokens (shape BATCHSIZE x SEQUENCE_LENGTH).
    '''
    lengths = [len(d['tokens']) for d in batch]
    maxlength = max(lengths)
    for d in batch:
        l = len(d['tokens'])
        d['attn_mask'] = [1] * len(d['tokens']) + [0] * (maxlength - l)
        d['tokens'] = list(d['tokens']) + [CONFIG.PAD_TOKEN_ID] * (maxlength - l)
    return {
        'attn_mask': torch.tensor([d['attn_mask'] for d in batch]),
        'tokens': torch.tensor([d['tokens'] for d in batch]),
        'labels': torch.tensor([list(d['labels']) for d in batch])
    }


class MyCheckpointCallback(Callback):
    ''' this saves a checkpoint after each epoch (each checkpoint 1.3GB). '''

    def __init__(self, path):
        self.epoch = 0
        self.path = path

    def on_train_epoch_end(self, trainer, pl_module):
        print("Training epoch done... calling callback.")
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        trainer.save_checkpoint(self.path + os.sep + '%s.cpt' % self.epoch, False)
        self.epoch += 1



def run_training(data_train, data_valid, class2int, epochs):
    model = RecipERT(len(class2int))
    dataset_train = RecipERTDataset(class2int, data_train)
    dataset_valid = RecipERTDataset(class2int, data_valid)
    dataloader_train = DataLoader(dataset=dataset_train,
                                  batch_size=CONFIG.BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=CONFIG.NUM_WORKERS,
                                  persistent_workers=True,
                                  collate_fn=collate_fn_pad)
    dataloader_valid = DataLoader(dataset=dataset_valid,
                                  batch_size=CONFIG.BATCH_SIZE,
                                  shuffle=False,
                                  num_workers=CONFIG.NUM_WORKERS,
                                  persistent_workers=True,
                                  collate_fn=collate_fn_pad)

    logger = WandbLogger(project='RecipERT')
    logger.watch(model, log='all')
    my_checkpoint_callback = MyCheckpointCallback('checkpoints')

    trainer = Trainer(logger=logger,
                      log_every_n_steps=1,
                      min_epochs=epochs,
                      max_epochs=epochs,
                      #gpus=[7], # Activate if needed (not tested on megagpu yet!)
                      callbacks=[my_checkpoint_callback])
    trainer.validate(model, dataloaders=dataloader_valid) # validate once before starting to train.
    trainer.fit(model, dataloader_train, dataloader_valid)

    return model




if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('USAGE: print_sample.py <training_json> <validation_json>')
        exit(1)

    with open(sys.argv[1], 'r') as f:
        data_train = json.load(f)
    with open(sys.argv[2], 'r') as f:
        data_valid = json.load(f)

    # first, train which recipe classes exist
    class2int = train_vocabulary_of_recipe_classes(data_train, CONFIG.NCLASSES)
    with open('class2int.pkl', 'wb') as f:
        pickle.dump(class2int, f)

    model = run_training(data_train, data_valid, class2int, epochs=CONFIG.EPOCHS)
