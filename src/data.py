"""
Data reading
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from collections import namedtuple

tf.flags.DEFINE_boolean('shuffle', False, 'Data shuffle')
tf.flags.DEFINE_integer('queue_capacity', 10000, 'Queue capacity')
tf.flags.DEFINE_integer('nthreads', 100, 'Number of threads')

FLAGS = tf.flags.FLAGS

SentenceBatch = namedtuple('SentenceBatch', ('ids', 'mask'))


def prefetch_input_data(reader,
                        file_pattern,
                        shuffle,
                        capacity,
                        num_reader_threads=1):
  """Prefetches string values from disk into an input queue.
  """
  data_files = []
  for pattern in file_pattern.split(','):
    data_files.extend(tf.gfile.Glob(pattern))
  if not data_files:
    tf.logging.fatal('Found no input files matching %s', file_pattern)
  else:
    tf.logging.info('Prefetching values from %d files matching %s',
                    len(data_files), file_pattern)

  filename_queue = tf.train.string_input_producer(
      data_files, shuffle=shuffle, capacity=16, name='filename_queue')

  if shuffle:
    min_after_dequeue = int(0.6 * capacity)
    values_queue = tf.RandomShuffleQueue(
        capacity=capacity,
        min_after_dequeue=min_after_dequeue,
        dtypes=[tf.string],
        shapes=[[]],
        name='random_input_queue')
  else:
    values_queue = tf.FIFOQueue(
        capacity=capacity,
        dtypes=[tf.string],
        shapes=[[]],
        name='fifo_input_queue')

  enqueue_ops = []
  for _ in range(num_reader_threads):
    _, value = reader.read(filename_queue)
    enqueue_ops.append(values_queue.enqueue([value]))
  tf.train.queue_runner.add_queue_runner(
      tf.train.queue_runner.QueueRunner(values_queue, enqueue_ops))
  tf.summary.scalar('queue/%s/fraction_of_%d_full' %
                    (values_queue.name, capacity),
                    tf.cast(values_queue.size(), tf.float32) * (1. / capacity))

  return values_queue


def parse_example_batch(serialized):
  """Parses a batch of Example protos.

  Args:
    serialized: A 1-D string Tensor; a batch of serialized Example protos.
  Returns:
    encode: Batch of encode sentences.
    decode_pre: Batch of "previous" sentences to decode.
    decode_post: Batch of "post" sentences to decode.
  """
  if FLAGS.multi_attr:
    features = {
        'sentence': tf.VarLenFeature(dtype=tf.int64),
        'attrib': tf.VarLenFeature(dtype=tf.int64),
        'label': tf.VarLenFeature(dtype=tf.int64)
    }
  else:
    features = {
        'sentence': tf.VarLenFeature(dtype=tf.int64),
        'label': tf.FixedLenFeature([], dtype=tf.int64),
    }

  features = tf.parse_example(serialized, features=features)

  def _sparse_to_batch(sparse):
    ids = tf.sparse_tensor_to_dense(sparse)
    mask = tf.sparse_to_dense(
        sparse.indices,
        sparse.dense_shape,
        tf.ones_like(
            sparse.values, dtype=tf.int8))
    return ids, mask, sparse.dense_shape

  ids, mask, shp = _sparse_to_batch(features['sentence'])

  if FLAGS.multi_attr:
    attrib_mask, _, _ = _sparse_to_batch(features['attrib'])
    labels, _, _ = _sparse_to_batch(features['label'])
    return ids, mask, shp, labels, attrib_mask
  return ids, mask, shp, features['label'], None

def data(file_pattern, batch_size):

  input_queue = prefetch_input_data(
      tf.TFRecordReader(),
      file_pattern,
      shuffle=FLAGS.shuffle,
      capacity=FLAGS.queue_capacity,
      num_reader_threads=FLAGS.nthreads)
  serialized = input_queue.dequeue_many(batch_size)
  sen = parse_example_batch(serialized)

  return sen
