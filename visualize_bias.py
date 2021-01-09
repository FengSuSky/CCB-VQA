import argparse
import json
import cPickle as pickle
from collections import defaultdict, Counter
from os.path import dirname, join
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, VQAFeatureDataset

from vqa_debias_loss_functions import *



def parse_args():
    parser = argparse.ArgumentParser("Train the BottomUpTopDown model with a de-biasing method")
    # Arguments we added
    parser.add_argument(
        '--cache_features', action="store_true",
        help="Cache image features in RAM. Makes things much faster, "
             "especially if the filesystem is slow, but requires at least 48gb of RAM")
    parser.add_argument(
        '--dataset', default='cpv2',
        choices=["v2", "cpv2", "cpv1"],
        help="Run on VQA-2.0 instead of VQA-CP 2.0"
    )
    parser.add_argument(
        '--mode', default="updn",
        choices=["updn", "q_debias","v_debias","q_v_debias"],
        help="Kind of ensemble loss to use")
    # Arguments from the original model, we leave this default, except we
    # set --epochs to 30 since the model maxes out its performance on VQA 2.0 well before then
    args = parser.parse_args()
    return args

def get_bias(train_dset,eval_dset):
    # Compute the bias:
    # The bias here is just the expected score for each answer/question type
    answer_voc_size = train_dset.num_ans_candidates

    # question_type -> answer -> total score
    question_type_to_probs = defaultdict(Counter)

    # question_type -> num_occurances
    question_type_to_count = Counter()
    for ex in train_dset.entries:
        ans = ex["answer"]
        q_type = ans["question_type"]
        question_type_to_count[q_type] += 1
        if ans["labels"] is not None:
            for label, score in zip(ans["labels"], ans["scores"]):
                question_type_to_probs[q_type][label] += score
    question_type_to_prob_array = {}

    for q_type, count in question_type_to_count.items():
        prob_array = np.zeros(answer_voc_size, np.float32)
        for label, total_score in question_type_to_probs[q_type].items():
            prob_array[label] += total_score
        prob_array /= count
        question_type_to_prob_array[q_type] = prob_array

    for ds in [train_dset,eval_dset]:
        for ex in ds.entries:
            q_type = ex["answer"]["question_type"]
            ex["bias"] = question_type_to_prob_array[q_type]
    return question_type_to_prob_array

def main():
    args = parse_args()
    dataset=args.dataset
    if dataset=='cpv1':
        dictionary = Dictionary.load_from_file('data/dictionary_v1.pkl')
    elif dataset=='cpv2' or dataset=='v2':
        dictionary = Dictionary.load_from_file('data/dictionary.pkl')

    print("Building train dataset...")
    train_dset = VQAFeatureDataset('train', dictionary, dataset=dataset,
                                   cache_image_features=args.cache_features)
    print("Building test dataset...")
    eval_dset = VQAFeatureDataset('val', dictionary, dataset=dataset,
                                  cache_image_features=args.cache_features)

    lable2answer = eval_dset.label2ans

    bias_p=get_bias(train_dset,eval_dset)
    bias_color=bias_p['what color is']

    bias_color_top5 = bias_color.argsort()[::-1][0:5]

    bias_color_p = []
    bias_color_word = []

    for i in bias_color_top5:
        bias_color_p.append(bias_color[i])
        bias_color_word.append(lable2answer[i])

    print(bias_color_p)
    print(bias_color_word)


if __name__ == '__main__':
    main()
