"""
Utils for vocabulary processing
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

FLAGS = tf.flags.FLAGS

class Vocab():

  def __init__(self):
    vocab_map = {}
    ivocab_map = {}
    lines = open(FLAGS.vocab_file, 'r').readlines()
    for i in range(len(lines)):
      word = lines[i].strip()
      vocab_map[word] = i
      ivocab_map[i] = word
    vocab_words = set(vocab_map.keys())

    self.vocab_map = vocab_map
    self.ivocab_map = ivocab_map
    self.vocab_words = vocab_words

  def convert_to_str(self, ids):
    st = ''
    for j in range(len(ids)):
      word_id = ids[j]
      # print(word_id)
      if word_id == 0:  # EOS
        break
      else:
        st += self.ivocab_map[word_id] + ' '
    return st

  def convert_to_inds(self, words):
    # ids = [self.vocab_map[word] for word in words]
    ids = []
    for word in words:
      if word in self.vocab_words:
        ids.append(self.vocab_map[word])
      elif word.lower() in self.vocab_words:
        ids.append(self.vocab_map[word.lower()])
      else:
        ids.append(self.vocab_map['<unk>'])
    # ids.append(0) #EOS
    return ids

  def construct_batch(self, sents, size=None):
    sents_ids = []
    for sent in sents:
      sents_ids.append(self.convert_to_inds(sent.split()))
    if not size:
      size = max([len(sen) for sen in sents_ids]) + 1  # include EOS
    n = len(sents)
    sen_batch = np.zeros((n, size), dtype=np.int32)
    msk_batch = np.zeros((n, size), dtype=np.int32)
    for i in range(n):
      sen = sents_ids[i]
      sen_batch[i][:min(len(sen), size)] = sen
      msk_batch[i][:min(len(sen)+1, size)].fill(1)

    shp = sen_batch.shape
    return sen_batch, msk_batch, shp

def add_loss(loss, losses):
  losses.append(loss)
  slim.losses.add_loss(loss[0])

def one_hot(labels, labels_mask=None):
  if FLAGS.multi_attr:
    sampled_labels = []
    for i in range(FLAGS.Nattr):
      sampled_labels_attr = tf.one_hot(labels[:,i], FLAGS.Nlabels[i])
      labels_mask_i = tf.reshape(labels_mask[:,i], (FLAGS.batch_size, 1))
      sampled_labels_attr *= tf.cast(labels_mask_i, tf.float32)
      sampled_labels_attr *= FLAGS.attrib_mask[i]
      sampled_labels.append(sampled_labels_attr)
    sampled_labels = tf.concat(sampled_labels, axis=1)
    return sampled_labels
  else:
    return tf.one_hot(labels, FLAGS.Nlabels)

def crop_batch(sen_batch, shp, batch_size=None):

  if not batch_size:
    batch_size = FLAGS.batch_size

  # This part is here because the tf.greater expects a tensor
  if FLAGS.custom_input:
    return tf.squeeze(tf.image.pad_to_bounding_box(
        tf.expand_dims(sen_batch, -1),
        0, 0, FLAGS.batch_size, FLAGS.input_size))
  else:
    sen_batch_fs = tf.cond(tf.greater(shp[1], FLAGS.input_size),
        lambda: tf.squeeze(tf.image.crop_to_bounding_box(
            tf.expand_dims(sen_batch, -1),
            0, 0, batch_size, FLAGS.input_size)),
        lambda: tf.squeeze(tf.image.pad_to_bounding_box(
            tf.expand_dims(sen_batch, -1),
            0, 0, batch_size, FLAGS.input_size)))
  return sen_batch_fs


def pad_if_smaller(batch, size):
  batch_fs = tf.cond(
      tf.greater(size, FLAGS.input_size),
      lambda: batch,
      lambda: tf.squeeze(tf.image.pad_to_bounding_box(
          tf.expand_dims(batch, -1), 0, 0, FLAGS.batch_size, FLAGS.input_size)),
  )
  return batch_fs


def proc(sen_batch, mask_batch, shp, labels_batch, batch_size=None):

  if FLAGS.crop_batches:
    sen_batch = crop_batch(sen_batch, shp, batch_size)
    mask_batch = crop_batch(mask_batch, shp, batch_size)
    batch_len = FLAGS.input_size
  else:
    batch_len = 0

  seq_len_encoder = tf.reduce_sum(mask_batch, axis=1)
  seq_len_decoder = seq_len_encoder
  labels_batch = tf.squeeze(labels_batch)

  return sen_batch, mask_batch, batch_len, seq_len_encoder, seq_len_decoder

def construct_eos_mask(eos_pred, num_steps):
  word_mask = 1 - tf.sign(tf.cumsum(tf.cast(eos_pred, tf.int32), axis=1))
  num_words = tf.reduce_sum(word_mask, axis=1)

  eos_mask = tf.sequence_mask(num_words+1, maxlen=num_steps, dtype=tf.float32)
  seq_len = num_words+1
  return eos_mask, seq_len
