"""
Convert corpus to TFRecord format
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import tensorflow as tf
from random import shuffle
from nltk.tokenize import word_tokenize

FLAGS = tf.flags.FLAGS

def get_sents(neg_path, label, vocab_list, vocab_map):
  with open(neg_path, 'r') as f:
    sents_read = f.readlines()
    sents = [sent.strip() for sent in sents_read]
    sent_cvt = []
    for sent in sents:
      word_ids = []
      count_unks = 0
      sent_tokenized = word_tokenize(sent)
      for word in sent_tokenized:
        word = word.lower()
        if word in vocab_list:
          word_ids.append(vocab_map[word])
        else:
          word_ids.append(vocab_map['<unk>'])
          count_unks += 1
      word_ids.append(vocab_map['<eos>'])
      if len(word_ids) >= 3*count_unks:  # Skip sents with majority unks
        sent_cvt.append((word_ids, label))
  return sent_cvt


def int64_feature(value):
  return tf.train.Feature(
      int64_list=tf.train.Int64List(value=[int(v) for v in value]))


def sort_by_len(data):
  data_lens = [len(dat[0]) for dat in data]
  sort_inds = np.argsort(data_lens)
  data = [data[i] for i in sort_inds]
  return data

def main():

  dataset = sys.argv[1]
  vocab_file = sys.argv[2]
  output_path = sys.argv[3]

  vocab_map = {}
  words = open(vocab_file, 'r').readlines()

  index = 0
  for word in words:
    vocab_map[word.strip()] = index
    index += 1
  vocab_set = set(vocab_map.keys())

  data = []

  for label, path in enumerate(dataset.split(',')):
    sents = get_sents(path, label, vocab_set, vocab_map)
    shuffle(sents)
    data.extend(sents)

  shuffle(data)

  train = data

  tfrecord_corpus_path_train = output_path + '/train.tfrecords'

  data_lists = [train]
  corpus_paths = [tfrecord_corpus_path_train]

  for data_list, corpus_path in zip(data_lists, corpus_paths):
    with tf.python_io.TFRecordWriter(corpus_path) as builder:
      index = 0
      for ex in data_list:
        word_ids = ex[0]
        label = ex[1]

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'sentence': int64_feature(word_ids),
                    'label': int64_feature([label])
                }))
        builder.write(example.SerializeToString())
        index += 1
        print(index)
    builder.close()

main()
