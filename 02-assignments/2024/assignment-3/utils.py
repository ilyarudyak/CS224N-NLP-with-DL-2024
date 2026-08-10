#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2022-23: Homework 4
utils.py: Utility Functions
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Vera Lin <veralin@stanford.edu>
Siyan Li <siyanli@stanford.edu>
Moussa KB Doumbouya <moussa@stanford.edu>
"""

from typing import List
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
import sentencepiece as spm
nltk.download('punkt')


def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
        The paddings should be at the end of each sentence.
    @param sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (str): padding token
    @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    sents_padded = []

    ### YOUR CODE HERE (~6 Lines)
    # Find the maximum length of the sentences
    max_length = max(len(sent) for sent in sents)

    # Pad each sentence to the maximum length
    for sent in sents:
        sent_padded = sent + [pad_token] * (max_length - len(sent))
        sents_padded.append(sent_padded)
    ### END YOUR CODE

    return sents_padded


def read_corpus(file_path, source, vocab_size=2500):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    @param vocab_size (int): number of unique subwords in
        vocabulary when reading and tokenizing
    """
    data = []
    sp = spm.SentencePieceProcessor()
    sp.load('{}.model'.format(source))

    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            subword_tokens = sp.encode_as_pieces(line)
            # only append <s> and </s> to the target sentence
            if source == 'tgt':
                subword_tokens = ['<s>'] + subword_tokens + ['</s>']
            data.append(subword_tokens)

    return data


def autograder_read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = nltk.word_tokenize(line)
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """

    # Calculate the number of batches
    batch_num = math.ceil(len(data) / batch_size)

    # Create an array of indices for the data
    # [0, 1, 2, ..., len(data)-1]
    # NOT indices from Vocabulary!
    index_array = list(range(len(data)))

    # If shuffle is True, randomly shuffle the indices
    if shuffle:
        np.random.shuffle(index_array)

    # Yield batches of source and target sentences
    for i in range(batch_num):

        # Get the indices for the current batch
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        # Get the examples for the current batch using the indices
        examples = [data[idx] for idx in indices]

        # Sort the examples in the batch by the length of the source sentence in descending order
        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)

        # Extract the source and target sentences from the examples
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents


