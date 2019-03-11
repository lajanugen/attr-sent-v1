"""
Sampling from the model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from model import model_sample
from data import data
from utils import Vocab
import tensorflow.contrib.slim as slim
import os

tf.flags.DEFINE_integer('batch_size', 128, 'batch size')

tf.flags.DEFINE_integer('vocab_size', 10000, 'vocab size')
tf.flags.DEFINE_integer('embedding_size', 300, 'word size')
tf.flags.DEFINE_integer('lstm_size', 500, 'lstm size')
tf.flags.DEFINE_integer('dec_lstm_size', 500, 'lstm size')
tf.flags.DEFINE_integer('input_size', 30, 'length of sentence')

tf.flags.DEFINE_boolean('crop_batches', False, 'Crop batches')
tf.flags.DEFINE_float('uniform_init_scale', 0.1, 'weights init scale')
tf.flags.DEFINE_boolean('load_embeddings', False, 'Load word embeddings')

tf.flags.DEFINE_string('mode', 'eval', 'Mode')
tf.flags.DEFINE_integer('decode_max_steps', 30, 'Max decode steps')
tf.flags.DEFINE_boolean('flip_input', False, 'Flip input order')
tf.flags.DEFINE_boolean('custom_input', False, 'Custom input')

tf.flags.DEFINE_boolean('multi_attr', False, 'Multiple attributes')
tf.flags.DEFINE_integer('Nattr', 1, 'Number of attributes')
tf.flags.DEFINE_integer('Nlabels', 2, 'Number of labels')
tf.flags.DEFINE_boolean('beam_search', False, 'Vocabulary transformation')

tf.flags.DEFINE_string('restore_ckpt_path', 'restore_path', 'restore path')
tf.flags.DEFINE_string('input_file', "", 'Vocab file list')
tf.flags.DEFINE_boolean('flip_label', False, 'Vocabulary transformation')
tf.flags.DEFINE_string('samples_dir', 'results/samples/', 'dir')
tf.flags.DEFINE_string('sampling_mode', 'max', 'dir')
tf.flags.DEFINE_string('vocab_file', '', 'Vocabulary file')
tf.flags.DEFINE_string('mdl_name', '', 'Name of model')

FLAGS = tf.flags.FLAGS
HOME_DIR = ''


def main(argv=()):
  del argv  # Unused.

  vocab = Vocab()

  shp_p = tf.placeholder(tf.int32, shape=(2,))
  sen_batch_p = tf.placeholder(tf.int32, shape=(FLAGS.batch_size, None))
  mask_batch_p = tf.placeholder(tf.int32, shape=(FLAGS.batch_size, None))
  labels_batch_p = tf.placeholder(tf.int32, shape=(FLAGS.batch_size,))

  max_sampling = (FLAGS.sampling_mode == 'max')
  decoded_samples = model_sample(
      sen_batch_p, mask_batch_p, shp_p, labels_batch_p, max_sampling=max_sampling)
  saver = tf.train.Saver()

  with tf.Session() as sess:
    coord = tf.train.Coordinator()

    saver.restore(sess, FLAGS.restore_ckpt_path)

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for label in range(FLAGS.Nlabels):
      if FLAGS.flip_label:
        flip_label = 1 - label
      else:
        flip_label = label
      input_file = FLAGS.input_file.split(',')[label]
      input_sents = open(input_file, 'r').readlines()

      input_sents = [sent.strip() for sent in input_sents]

      samples = []
      for it in range(int(len(input_sents)/FLAGS.batch_size)+1):

        labels_batch = np.array([0]*FLAGS.batch_size)
        sents = input_sents[it*FLAGS.batch_size: (it+1)*FLAGS.batch_size]
        num_sents = len(sents)
        while len(sents) < FLAGS.batch_size:
          sents.extend(sents[:FLAGS.batch_size - len(sents)])
        sen_batch, mask_batch, shp = vocab.construct_batch(sents)

        out = sess.run(decoded_samples,
            feed_dict={sen_batch_p:sen_batch, mask_batch_p:mask_batch, shp_p:shp, labels_batch_p:labels_batch})

        for k in range(FLAGS.batch_size):
          if k >= num_sents:
            break
          samples.append(vocab.convert_to_str(out[flip_label][k]))

      fname = FLAGS.samples_dir + '/' + FLAGS.mdl_name + '_sample_' + str(flip_label)
      fname += '.txt'
      with open(fname, 'w') as results_file:
        results_file.write('\n'.join(samples))

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
  main()
