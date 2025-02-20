from collections import defaultdict

import numpy as np
import torch
import time
import jsonlines
import sys
import gc

from sklearn.metrics import f1_score

from config import get_model_and_tokenizer
import config



def tokenize(text, tokenizer):
    result = tokenizer(text)
    tokens = result['input_ids']
    x = torch.tensor(tokens)
    if (x.ndim==1):
        x = x.unsqueeze(0)
    return x


def compute_log_likelihood(model, tokenizer, text, prompt):
    """
    Compute the log-likelihood of a text using teacher forcing.

    Args:
        model: The language model (e.g., GPT2LMHeadModel).
        tokenizer: Tokenizer corresponding to the model.
        text: Input text as a string.

    Returns:
        log_likelihood: Normalized log-likelihood of the text.
    """
    # Tokenize the text
    tokenized_prompt = tokenize(prompt, tokenizer).to(model.device)
    prompt_length = tokenized_prompt.size(1)
    tokenized_text = tokenize(text, tokenizer).to(model.device)
    #print("TT: ", tokenized_text)
    n = tokenized_text.size(1)  # Length of the sequence

    # Get logits from the model
    with torch.no_grad():
        output = model(tokenized_text, labels=tokenized_text)
        logits = output.logits  # [batch_size, seq_len, vocab_size]

    #print("LOGITS: ", logits)

    # Calculate probabilities for each token
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)  # Log-softmax over vocabulary

    # Extract log-probabilities for the actual tokens in the input text
    # Shift the input tokens to align with predictions
    actual_token_indices = tokenized_text[:, prompt_length+1:]  # True tokens, ignoring the first
    true_token_log_probs = probabilities[0, prompt_length:-1, :].gather(1, actual_token_indices.T)  # [seq_len-1, 1]
    #print("TTP: ", true_token_log_probs)

    # Compute normalized log-likelihood
    log_likelihood = true_token_log_probs.mean().item()  # Normalize by number of tokens
    #log_likelihood = true_token_log_probs.sum().item() / (n - 1)  # Normalize by number of tokens
    return log_likelihood

def eval_which_lm(log_likelihoods, ground_truth):
    predicted_models = {i: max(lls, key=lls.get) for i, lls in log_likelihoods.items()}
    print("Predicted Models: ", predicted_models)

    # Initialize accuracy tracking for each model
    model_accuracies = defaultdict(int)
    total_samples = defaultdict(int)

    # Compute per-model accuracy
    for i in predicted_models:
        total_samples[ground_truth[i]] += 1
        if predicted_models[i] == ground_truth[i]:
            model_accuracies[ground_truth[i]] += 1

    # Calculate percentages
    model_accuracy_percentages = {model: (correct / total_samples[model]) * 100
                                  for model, correct in model_accuracies.items()}

    # Output accuracy for each model
    print("Accuracy for each model:")
    for model, accuracy in model_accuracy_percentages.items():
        print(f"{model}: {accuracy:.2f}%")

    # Find the model with the highest accuracy
    best_model = max(model_accuracy_percentages, key=model_accuracy_percentages.get)
    best_accuracy = model_accuracy_percentages[best_model]
    print(f"Model with the highest accuracy: {best_model} ({best_accuracy:.2f}%)")

    accuracy = sum(1 for i in predicted_models if predicted_models[i] == ground_truth[i]) / len(predicted_models) * 100
    print('-------------')
    print(f"Overall Accuracy: {accuracy:.2f}%")

def eval_human_or_lm(log_likelihoods, ground_truth):
    mapped_likelihoods = [i[max(i, key=i.get)] for i in log_likelihoods.values()]
    mapped_likelihoods = np.array(mapped_likelihoods)


    is_human = mapped_likelihoods < config.IS_HUMAN_THRESHOLD
    print("human: ", is_human)
    # Drop keys, keep only values
    ground_truth_values = np.array(list(ground_truth.values()))
    print("ground_truth_values: ", ground_truth_values)
    # Create a boolean array where True means 'human' and False otherwise
    true_classes = ground_truth_values == 'human'
    print("1. true_classes ", true_classes)

    f1 = f1_score(true_classes, is_human)

    print(f"F1 score: {f1:.2f}")


if __name__ == '__main__':
    # You can use this text to play around with the token probabilities
    #text = "The apple does not fall far from the tree"

    INPUT_FILE = sys.argv[1]    # this is the input .jsonl file

    # setup
    log_likelihoods = {}
    ground_truth = {}
    texts = {}
        
    for key in config.KEYS:

        # load the next language model
        print('Setting up model', key, '...')
        start = time.time()
        model, tokenizer = get_model_and_tokenizer(key)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        print('Done. Took %.2f seconds.' %(time.time()-start))
        
        with torch.no_grad():
            #log_likelihood = compute_log_likelihood(model, tokenizer, text)
            #print(log_likelihood)
            #print("---------------------")

            with jsonlines.open(INPUT_FILE) as reader:

                # loop over input texts
                for i,(text,prompt,label) in enumerate(reader):

                    # FIXME:
                    # 1. tokenize text with tokenizer
                    # 2. run tokens through model, get logits
                    # 3. !!! compute log_likelihood !!!
                    log_likelihood = compute_log_likelihood(model, tokenizer, text, prompt)

                    if i not in log_likelihoods:
                        log_likelihoods[i] = {}
                        ground_truth[i] = label
                        texts[i] = text

                    # store log_likelihood
                    log_likelihoods[i][key] = log_likelihood


        # freeing GPU memory for the next model
        del tokenizer
        del model
        gc.collect()
        torch.cuda.empty_cache()

        
    # 4. evaluate results: for each model, compare if model with highest log likelihood
    #    equals the true model (see 'label') which generated the text.
    #
    #    For Experiment 1 (which-lm), compute the accuracy, i.e. the percentage
    #                  of correctly chosen models.
    #    For Experiment 2 (lm-vs-human), compute the f1 score
    #                  (you can use sklearn.metrics.f1_score).
    #print(())

    #eval_which_lm(log_likelihoods, ground_truth)
    eval_human_or_lm(log_likelihoods, ground_truth)





    pass

    

        
