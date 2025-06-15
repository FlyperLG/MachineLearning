NCLASSES            = 50                            # number of recipe classes for your classifier

''' transformer parameters '''
MODEL_ID            = 'bert-base-german-cased'      # use this pre-trained transformer as your backbone.
PAD_TOKEN_ID        = 1                             # the transformer pads with tokens of this ID.
D                   = 768                           # dimensionality of the transformer's embeddings.
MAX_LENGTH          = 512                           # maximum sequence length accepted by the transformer.

''' training parameters'''
BATCH_SIZE          = 20
EPOCHS              = 10
LR                  = 0.000006
NUM_WORKERS         = 16                             # how many processes in parallel are preparing batches
                                                    # for the GPU. You might want to increase this if your
                                                    # GPU idles while your CPUs are busy.


''' feel free to add further parameters ... '''
PROB_THRESHOLD      = 0.4