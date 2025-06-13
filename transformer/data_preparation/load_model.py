#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script loads trained language models and extracts surprisal values.
Modified to handle GPU memory issues and tensor size mismatches.
"""

import argparse
import torch
import sys
import pickle
import numpy as np
import pandas as pd
import os
from collections import defaultdict

# Add your project functions path
sys.path.append('/content/language_models/transformer/functions')
from encoders import nwp_transformer
from prep_text import word_2_index

# Argument parser
parser = argparse.ArgumentParser(description='Get file name')
parser.add_argument('-data_loc', type=str, default='/content/language_models/wiki_validation_pos.txt')
parser.add_argument('-model_loc', type=str, default='/content/language_models/transformer/nwp_model_1_11711340')
parser.add_argument('-output_id', type=str, default='surprisal_output')
parser.add_argument('-dict_loc', type=str, default='/content/language_models/transformer/wiki_train_word_pos_indices')
parser.add_argument('-output_file', type=str, default='/content/language_models/transformer/surprisal_output.csv')
args = parser.parse_args()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)

nwp_dict = load_obj(args.dict_loc)
dict_size = len(nwp_dict) + 1

def index_2_word(dictionary, indices):
    rev_dictionary = defaultdict(str)
    for x, y in dictionary.items():
        rev_dictionary[y] = x
    return [[rev_dictionary[int(i)] for i in ind] for ind in indices]

def calc_surprisal(data_loc, model, batch_size=8):
    from torch.nn.utils.rnn import pad_sequence

    sent = []
    with open(data_loc) as file:
        for line in file:
            sent.append(['<s>'] + line.strip().split() + ['</s>'])

    original_sent = [s[1:] for s in sent]

    sent_idx, lengths = word_2_index(sent, len(sent), nwp_dict)

    # Convert to tensors
    sent_idx = [torch.tensor(seq, dtype=torch.long) for seq in sent_idx]
    lengths = torch.tensor(lengths, dtype=torch.long)

    all_surprisals = []
    all_targets = []
    all_original = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(sent_idx), batch_size):
            batch_sents = sent_idx[i:i + batch_size]
            batch_lengths = lengths[i:i + batch_size]
            batch_original = original_sent[i:i + batch_size]

            padded_sents = pad_sequence(batch_sents, batch_first=True, padding_value=0).float().to(device)
            batch_lengths = batch_lengths.to(device)

            predictions, targets = model(padded_sents, batch_lengths)

            surprisal = -torch.log_softmax(predictions, dim=2).squeeze()
            surprisal = surprisal.gather(-1, targets.unsqueeze(-1)).squeeze()

            surprisal = surprisal.detach().cpu().numpy()
            targets = targets.cpu().numpy()
            batch_lengths = batch_lengths.cpu().numpy()

            for s, t, l, orig in zip(surprisal, targets, batch_lengths, batch_original):
                all_surprisals.append(s[:l - 1])
                all_targets.append(t[:l - 1])
                all_original.append(orig[:l - 1])

    return all_surprisals, all_targets, all_original

def clean_surprisal(surprisal, targets, original_sent):
    keep = [[not np.isnan(val) for val in s] for s in surprisal]
    surprisal = [[s[i] for i in range(len(s)) if k[i]] for s, k in zip(surprisal, keep)]
    targets = [[t[i] for i in range(len(t)) if k[i]] for t, k in zip(targets, keep)]
    original_sent = [[w[i] for i in range(len(w)) if k[i]] for w, k in zip(original_sent, keep)]

    rows = [(i+1, j+1, val) for i, sent in enumerate(surprisal) for j, val in enumerate(sent)]
    words = [w for sent in targets for w in sent]
    orig_words = [w for sent in original_sent for w in sent]

    df = pd.DataFrame(rows, columns=["sent_nr", "word_pos", "surprisal"])
    df["item"] = df.sent_nr * 100 + df.word_pos
    df["word"] = index_2_word(nwp_dict, [[w] for w in words])
    df["original word"] = orig_words
    return df

# Model configuration
model_config = {
    'embed': {'n_embeddings': dict_size, 'embedding_dim': 400, 'sparse': False, 'padding_idx': 0},
    'tf': {'in_size': 400, 'fc_size': 1024, 'n_layers': 2, 'h': 8, 'max_len': 128},  # Increased max_len for safety
    'cuda': torch.cuda.is_available()
}

# Load model
model = nwp_transformer(model_config)
model.load_state_dict(torch.load(args.model_loc, map_location=device))
model.to(device)
model.eval()

# Calculate and save
surprisal, targets, original_sent = calc_surprisal(args.data_loc, model)
df = clean_surprisal(surprisal, targets, original_sent)
df["surprisal"] = df["surprisal"].round(4)
df.sent_nr = df.sent_nr.astype(int)
df.word_pos = df.word_pos.astype(int)
df.to_csv(args.output_file, index=False)
# print(f"Saved surprisal values to {args.output_file}")
