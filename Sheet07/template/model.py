import torch.nn
import wandb
from pytorch_lightning import LightningModule
from torch.optim import Adam
from transformers import AutoModel

import config as CONFIG



class RecipERT(LightningModule):
    '''
    your model, aka "RecipERT". RecipERT ist a multi-label classifier which - given a recipe -
    returns multiple categories for the recipe.
    '''

    def __init__(self, ncategories):
        super(RecipERT, self).__init__()
        self.transformer = AutoModel.from_pretrained(CONFIG.MODEL_ID)
        self.head_layer = torch.nn.Linear(self.transformer.config.hidden_size, ncategories)
        self.loss = torch.nn.BCELoss()

    def forward(self, tokens, attn_mask, return_logits=True):
        '''
        your model's forward pass.
        @param tokens: a batch of tokens, i.e. a tensor of shape BATCHSIZE x SEQUENCELENGTH
        @type tokens: torch.tensor(long)
        @attn_mask: the attention mask for feeding to the transformer
                    (required to ignore padding tokens in attention).
        @type attn_mask: torch.tensor(long)
        @param return_logits: a flag indicating if the model should return logits (logits=True)
                       or probabilities (logits=False).
        @type return_logits: boolean
        '''
        # FIXME
        embeddings = self.transformer(tokens, attn_mask).last_hidden_state
        cls_embeddings = embeddings[:, 0, :]

        logits = self.head_layer(cls_embeddings)

        if return_logits:
            return logits
        else:
            return torch.nn.functional.sigmoid(logits)


    def training_step(self, batch, batch_idx):
        '''
            implement the computation of the loss for a training batch here.
            @param batch: a batch of training samples, of the form

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
        tokens,labels,attn_mask = batch['tokens'],batch['labels'],batch['attn_mask']
        result = self.forward(tokens, attn_mask, return_logits=False)
        loss = self.loss(result, labels.float())

        print('Train_loss', loss)

        # Log the training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)  # Lightning internal logging
        wandb.log({'train_loss': loss.item()}, step=self.global_step)
        # FIXME
        # remark: optionally, you can log the loss to tensorboard or wandb here.
        return loss


    def validation_step(self, batch, batch_idx):
        '''
            implement the computation of the loss for a validation batch here (see parameters
            above, in 'training step').
        '''
        tokens,labels,attn_mask = batch['tokens'],batch['labels'],batch['attn_mask']
        result = self.forward(tokens, attn_mask, return_logits=False)
        loss = self.loss(result, labels.float())

        print('val_loss', loss.item())
        # Log the validation loss
        self.log('valid_loss', loss, on_step=False, on_epoch=True, logger=True)  # Lightning internal logging
        wandb.log({'valid_loss': loss.item()}, step=self.global_step)
        # FIXME
        # remark: optionally, you can log the loss to tensorboard or wandb here.
        return loss


    def test_step(self, batch, batch_idx):
        raise NotImplementedError()


    def configure_optimizers(self):
        return Adam(self.parameters(), lr=CONFIG.LR)

