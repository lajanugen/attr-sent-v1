from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

FLAGS = tf.flags.FLAGS

class adv_cls_cgan(object):

  def __init__(self, cnt_enc, embedding_size, input_labels, labels, reuse=False):

    with tf.variable_scope('sen_pred', reuse=reuse):
      sen_pred = slim.fully_connected(
        cnt_enc, 1, activation_fn=None)

    with tf.variable_scope('attr_pred', reuse=reuse):
      attr_emb = slim.fully_connected(
        input_labels, embedding_size, activation_fn=None)
      attr_pred = tf.reduce_sum(attr_emb * cnt_enc, axis=1)

    pred = tf.squeeze(sen_pred) + tf.squeeze(attr_pred)

    self.scores = pred

    self.predictions = tf.cast(tf.greater(self.scores, 0.5), tf.int64)
  
    self.scores_probs = tf.nn.sigmoid(self.scores)
    labels = tf.cast(labels, tf.float32)
    losses = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self.scores, labels=labels)
  
    # CalculateMean cross-entropy loss
    self.pred_prob = self.scores_probs
    self.losses = losses
    self.loss = tf.reduce_mean(losses)

    # Accuracy
    self.correct_predictions = tf.equal(self.predictions, tf.cast(labels, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, 'float'))
     
def RNN_enc(sen_batch, embedding_size, batch_size,
    reuse=False, mask_zeros=False, mask_batch=None):

  # Placeholders for input, output and dropout
  embedded_chars = sen_batch

  # Keeping track of l2 regularization loss (optional)
  if mask_zeros:
    mask_exp = tf.expand_dims(tf.cast(mask_batch, tf.float32), -1)
    mask_tile = tf.tile(mask_exp, [1, 1, embedding_size])
    embedded_chars *= mask_tile

  cell_fw = tf.contrib.rnn.GRUCell(num_units=FLAGS.lstm_size//2, reuse=reuse)
  cell_bw = tf.contrib.rnn.GRUCell(num_units=FLAGS.lstm_size//2, reuse=reuse)

  zero_state = tf.zeros((batch_size, embedding_size))
  seq_len = tf.reduce_sum(tf.cast(mask_batch, tf.int64), axis=1)

  outputs, states = tf.nn.bidirectional_dynamic_rnn(
      cell_fw=cell_fw, cell_bw=cell_bw, dtype=tf.float32, 
      inputs=embedded_chars, sequence_length=seq_len)
  cnt_enc = tf.concat(states, 1)

  return cnt_enc

def classifier_enc(sen_batch, embedding_size, batch_size,
    mask_zeros, mask_batch):

  real = RNN_enc(sen_batch[0], embedding_size, batch_size,
      reuse=False, mask_zeros=mask_zeros, mask_batch=mask_batch[0])
  fake = RNN_enc(sen_batch[1], embedding_size, batch_size,
      reuse=True, mask_zeros=mask_zeros, mask_batch=mask_batch[1])
  rep_size = FLAGS.lstm_size

  return real, fake, rep_size

def cgan_classifier(sen_batch, labels, embedding_size, 
      batch_size, mask_zeros=False, 
      mask_batch=None, labels_mask=None):

  real, fake, rep_size = classifier_enc(sen_batch, embedding_size,
      batch_size, mask_zeros, mask_batch)

  real_labels = labels[0]
  fake_labels = labels[1]
  
  rf_labels = tf.ones((FLAGS.batch_size, ))
  cls_adv_real = adv_cls_cgan(
    real,
    embedding_size=rep_size,
    input_labels=real_labels,
    labels=rf_labels,
    reuse=False)

  rf_labels = tf.zeros((FLAGS.batch_size, ))
  cls_adv_fake = adv_cls_cgan(
    fake,
    embedding_size=rep_size,
    input_labels=fake_labels,
    labels=rf_labels,
    reuse=True)
    
  rf_labels = tf.zeros((FLAGS.batch_size, ))
  cls_adv_match = adv_cls_cgan(
    real,
    embedding_size=rep_size,
    input_labels=fake_labels,
    labels=rf_labels,
    reuse=True)
    
  adv_loss = 0.5 * cls_adv_real.loss + 0.25*(cls_adv_fake.loss + cls_adv_match.loss)
  adv_acc = 0.5 * (cls_adv_real.accuracy + cls_adv_fake.accuracy)

  gen_adv_losses = {}
  gen_adv_losses['rf'] = {'adv_loss': adv_loss, 'adv_acc': adv_acc, 'gen_loss': -cls_adv_fake.loss}

  return gen_adv_losses
