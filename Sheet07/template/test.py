
import sys
import json
import pickle
import torch
from transformers import AutoTokenizer
import config as CONFIG

from template.model import RecipERT

# Check device
def get_device():
    if torch.cuda.is_available():
        print('CUDA available')
        return 'cuda'
    if torch.backends.mps.is_available():
        print('MPS available')
        return 'mps'
    else:
        return 'cpu'


def apply_model(model, class2int, data_test):

    # FIXME
    # load test data
    # load class2int
    # for each test recipe:
    #   apply the model, obtaining scores
    #   threshold the scores at 40%, obtaining a set of predicted classes
    #   measure f1 between the estimated and true classes
    # report the f1, averaged over all test recipes.
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.MODEL_ID)
    device = get_device()
    scores = []

    for recipe in data_test:
        text = recipe['instructions']
        categories = recipe['recipeCategory']

        int2class = {v: k for k, v in class2int.items()}

        # Tokenize the input
        encoded_input = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=CONFIG.MAX_LENGTH,  # Ensure this matches the training configuration
            return_tensors="pt",
        )
        tokens = encoded_input["input_ids"].to(device)
        attn_mask = encoded_input["attention_mask"].to(device)

        with torch.no_grad():  # Disable gradient calculation for inference
            result = model(tokens, attn_mask, return_logits=False)[0]

        # Threshold probabilities at 40%
        predicted_classes = [i for i, prob in enumerate(result) if prob >= CONFIG.PROB_THRESHOLD]
        predicted_labels = [int2class[idx] for idx in predicted_classes]
        print(categories)
        print(predicted_labels)


        # Calculate precision, recall, and F1-score
        true_classes = set(categories)
        predicted_classes_set = set(predicted_labels)

        intersection = true_classes & predicted_classes_set
        print(intersection)
        print("_________")
        recall = len(intersection) / (len(true_classes) + 1e-8)
        precision = len(intersection) / (len(predicted_classes_set) + 1e-8)
        f1_score = (2 * precision * recall) / (precision + recall + 1e-8)

        scores.append({'recall': recall, 'precision': precision, 'f1_score': f1_score})

    # Calculate and display final averaged scores
    avg_recall = sum([score['recall'] for score in scores]) / len(scores)
    avg_precision = sum([score['precision'] for score in scores]) / len(scores)
    avg_f1_score = sum([score['f1_score'] for score in scores]) / len(scores)

    print(f"Averaged Recall: {avg_recall:.4f}")
    print(f"Averaged Precision: {avg_precision:.4f}")
    print(f"Averaged F1-score: {avg_f1_score:.4f}")

    print(scores)


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('USAGE: print_sample.py <model_checkpoint> <class2int.pkl> <test_path>')
        exit(1)

    with open(sys.argv[3], 'r') as f:
        data_test = json.load(f)
    with open(sys.argv[2], 'rb') as f:
        class2int = pickle.load(f)


    checkpoint_path = sys.argv[1]
    model = RecipERT.load_from_checkpoint(checkpoint_path, ncategories=len(class2int))
    model.eval()

    apply_model(model, class2int, data_test)

