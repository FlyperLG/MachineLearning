from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import GPT2LMHeadModel

HF_CACHE_DIR = '/data/stud/ML2425/team6/.cache'

# how the dataset was generated
DATASET_MAX_LENGTH = 100
DATASET_N_SEQUENCES_PER_PROMPT = 1
DATASET_GENERATION_STRATEGY = 'beamsearch-multinomial'

IS_HUMAN_THRESHOLD = -1.50

# which models to use
KEYS = ['gpt2',
        'gpt2-large',
        'falcon7b-instruct',
        'falcon7b',
        'codegen'
        ]


# if all LM's likelihood scores are below this threshold, we'll categorize the text as human.
CLASSIFIER_HUMAN_THRESH = None # FIXME: pick a threshold for the second experiment


'''
PROMPTS = [
    #"Adrian Ulges is a",
    "Green Sponarch is",
    #"Barbara St Poopbury is",
    #"'Strawberry Abyss'",
    #"import torch.nn as nn\n\nclass MyMLP(nn.Module):\n",
    #"The pellet with the poison's in the",
    #"Please describe Brian Hood's involvement in the scandal",
    #"Dunkel liegt die Stadt.",
    #"Ich erkläre dir wie man Schweißbahnen verlegt",
    #"The best strategy to win with HRE in AOE4 is",
    #"To the well-organized mind,"
    ]
'''


def get_model_and_tokenizer(key):
    if key == 'falcon7b-instruct':
        return get_model_and_tokenizer_default('tiiuae/falcon-7b-instruct')
    elif key == 'falcon7b':
        return get_model_and_tokenizer_default('tiiuae/falcon-7b')
    elif key == 'gpt2':
        return get_model_and_tokenizer_gpt2('gpt2')
    elif key == 'gpt2-large':
        return get_model_and_tokenizer_gpt2('gpt2-large')
    elif key == 'codegen': 
        return get_model_and_tokenizer_default('Salesforce/codegen-350M-mono')
    else:
        print('Unknown model key: %s. Exit.' %(key))
        exit(1)


def get_model_and_tokenizer_default(model_id):
    ''' this method returns the model+tokenizer for non-GPT2 models.'''
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=HF_CACHE_DIR)
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=HF_CACHE_DIR)
    return model,tokenizer

def get_model_and_tokenizer_gpt2(model_id):
    ''' this method returns the model+tokenizer for the GPT-2 model.'''
    model = GPT2LMHeadModel.from_pretrained(model_id, cache_dir=HF_CACHE_DIR)
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=HF_CACHE_DIR)
    return model,tokenizer
