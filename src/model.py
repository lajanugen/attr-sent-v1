""" Model implementation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from adv_cls import cgan_classifier
from utils import add_loss
from utils import construct_eos_mask
from utils import one_hot
from utils import proc
from tensorflow.python.layers import core as layers_core
from beam_search_decoder_v10 import BeamSearchDecoder
import itertools

FLAGS = tf.flags.FLAGS

def decode_beam(cnt, labels_oh, reuse=False,
           batch_size=None, decode_num_steps=None):

  if not batch_size:
    batch_size = FLAGS.batch_size

  if not decode_num_steps:
    decode_num_steps = FLAGS.decode_max_steps

  initial_state = cnt
  end_token = 0
  beam_width = 10
  word_vectors = slim.get_variables_by_name('rnn/word_embedding')[0]
  decoder_cell = tf.contrib.rnn.GRUCell(num_units=FLAGS.dec_lstm_size,
                                        reuse=reuse)

  sos_emb = sos(labels_oh, reuse=reuse)
  sos_emb = tf.tile(tf.expand_dims(sos_emb, 1), (1, beam_width, 1))
  with tf.variable_scope('classifier', reuse=reuse):
    projection_layer = layers_core.Dense(FLAGS.vocab_size, name="classifier")
  initial_state = tf.contrib.seq2seq.tile_batch(
      initial_state, multiplier=beam_width)

  # Define a beam-search decoder
  with tf.variable_scope('rnn', reuse=reuse):
    decoder = BeamSearchDecoder(
          cell=decoder_cell,
          embedding=word_vectors,
          start_tokens=sos_emb,
          end_token=end_token,
          initial_state=initial_state,
          beam_width=beam_width,
          output_layer=projection_layer,
          length_penalty_weight=0.0)

    outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
          decoder,
          maximum_iterations=FLAGS.decode_max_steps,
          swap_memory=True)
          #scope=decoder_scope)
    predictions = outputs.predicted_ids[:, :, 0]

  return predictions, None

def sos(labels_oh, reuse=None):
  if reuse:
    sos_emb = slim.get_variables_by_name('dec/sos')[0]
  else:
    sos_emb = tf.get_variable('sos', shape=[1, FLAGS.embedding_size])
  sos_emb = tf.tile(sos_emb, [FLAGS.batch_size, 1])
  return sos_emb

def decode(cnt, labels_oh, reuse=False,
           src_states=None, src_mask=None,
           max_sampling=True,
           batch_size=None, decode_num_steps=None,
           pred_logprobs=False):

  if not batch_size:
    batch_size = FLAGS.batch_size
  if not decode_num_steps:
    decode_num_steps = FLAGS.decode_max_steps

  if FLAGS.beam_search:
    return decode_beam(cnt, labels_oh, reuse=reuse, batch_size=batch_size,
	decode_num_steps=decode_num_steps)

  word_vectors = slim.get_variables_by_name('rnn/word_embedding')[0]

  initial_state = tf.zeros([FLAGS.batch_size, FLAGS.dec_lstm_size])
  initial_state = cnt

  states = []
  all_predictions = []
  all_predictions_mask = []
  all_pred_logprobs = []

  state = initial_state
  word_inds = None
  predictions_embs = None

  for i in range(decode_num_steps):
    if i == 0:
      embedding_input = sos(labels_oh, reuse=reuse)
    else:
      embedding_input = tf.nn.embedding_lookup(word_vectors, word_inds)

    decoder_cell = tf.contrib.rnn.GRUCell(num_units=FLAGS.dec_lstm_size,
                                          reuse=(i > 0) or reuse)
    decoder_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_cell,
        input_keep_prob=FLAGS.dropout_keep_prob_dec)
    with tf.variable_scope('rnn', reuse=(i > 0) or reuse):
      output, state = decoder_cell(embedding_input, state)

    states.append(tf.expand_dims(state, 1))

    predictions_logits = slim.fully_connected(
        output, FLAGS.vocab_size, activation_fn=None, scope='classifier',
        reuse=(i > 0) or reuse)

    predictions = tf.nn.softmax(predictions_logits)

    if max_sampling:
      word_inds = tf.argmax(predictions, axis=1)
    else:
      word_inds = tf.squeeze(tf.multinomial(predictions_logits, 1))

    if pred_logprobs:
      lp = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=predictions_logits, labels=word_inds)
      all_pred_logprobs.append(tf.expand_dims(lp, -1))

    word_inds_rshp = tf.expand_dims(word_inds, -1)
    mask = tf.equal(word_inds_rshp, 0)
    all_predictions.append(word_inds_rshp)
    all_predictions_mask.append(mask)

  states = tf.concat(states, axis=1)

  predictions = tf.concat(all_predictions, axis=1)
  predictions_mask = tf.concat(all_predictions_mask, axis=1)
  predictions_mask, predictions_seq_len = construct_eos_mask(predictions_mask, decode_num_steps)
  predictions = tf.cast(tf.cast(predictions, tf.float32) * predictions_mask,
                        tf.int64)
  if pred_logprobs:
    pred_lps = tf.concat(all_pred_logprobs, axis=1) * predictions_mask
    return predictions, predictions_mask, pred_lps, predictions_seq_len, states
  return predictions, predictions_mask


def classify(dec_outputs, targets, mask, labels_oh, reuse=False,
             src_states=None, src_mask=None, initial_state=None,
             attention_lm='def', batch_size=None):

  if not batch_size:
    batch_size = FLAGS.batch_size

  pred_input_flat = tf.reshape(dec_outputs, [-1, FLAGS.dec_lstm_size])

  mask = tf.cast(mask, tf.float32)

  predictions_logits = slim.fully_connected(
      pred_input_flat, FLAGS.vocab_size, activation_fn=None, scope='classifier',
      reuse=reuse)

  mask_flat = tf.reshape(mask, [-1])
  labels = tf.reshape(targets, [-1])

  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=predictions_logits, labels=labels)

  logprobs = tf.nn.log_softmax(predictions_logits)
  targets_mask = tf.one_hot(tf.reshape(labels, [-1]), FLAGS.vocab_size,
          dtype=tf.bool, on_value=True, off_value=False)
  logprobs = tf.boolean_mask(logprobs, targets_mask)
  logprobs *= mask_flat
  logprobs = tf.reshape(logprobs, (FLAGS.batch_size, -1))
  logprobs = tf.reduce_sum(logprobs, axis=1)
        
  masked_loss_flat = loss * mask_flat

  masked_loss = tf.reshape(masked_loss_flat, [batch_size, -1])

  mean_loss_by_example = tf.reduce_sum(masked_loss, 1)
  mean_loss = tf.reduce_mean(mean_loss_by_example)

  return mean_loss

def get_decoder_input_state(embs, cnt_enc, labels_oh, batch_len, batch_size=None):

  if not batch_size:
    batch_size = FLAGS.batch_size

  sos_emb = sos(labels_oh)
  sos_emb = tf.expand_dims(sos_emb, axis=1)
  decoder_input_embs = tf.concat([sos_emb, embs[:, :-1, :]], axis=1)

  dec_input = decoder_input_embs
  initial_state = cnt_enc

  return dec_input, initial_state

def flip_input(embs, lens):
  if FLAGS.flip_input:
    embs_flip = tf.reverse_sequence(
        embs, batch_axis=0, seq_axis=1, seq_lengths=tf.cast(lens, tf.int32))
  else:
    embs_flip = embs
  return embs_flip

def mult_int(enc, labels_oh, reuse=None):

  assert FLAGS.dec_lstm_size > FLAGS.lstm_size
  sty_prj = slim.fully_connected(
      labels_oh, FLAGS.dec_lstm_size - FLAGS.lstm_size, activation_fn=None,
      scope='cnt_sty', reuse=reuse)
  cnt_sty = tf.concat((enc, sty_prj), axis=1)
  return cnt_sty

def sample_labels_single_attr(labels, Nlabels):

  if Nlabels == 2:
    return 1 - labels

  labels = tf.squeeze(labels)
  label_inds = tf.tile(tf.expand_dims(tf.range(Nlabels), 0),
                       (FLAGS.batch_size, 1))
  oh = tf.one_hot(labels, Nlabels)
  mask = 1 - oh
  mask_bool = tf.cast(mask, tf.bool)
  mask_bool = tf.reshape(mask_bool, (FLAGS.batch_size, Nlabels))
  label_mask_out = tf.boolean_mask(label_inds, mask_bool)
  other_labels = tf.reshape(label_mask_out, (FLAGS.batch_size, -1))
  rand_inds = tf.random_uniform((FLAGS.batch_size, 1), maxval=Nlabels-1,
                                dtype=tf.int32)
  range_inds = tf.expand_dims(tf.range(FLAGS.batch_size), -1)
  label_pick_inds = tf.squeeze(tf.stack((range_inds, rand_inds), axis=1))

  sampled_labels = tf.gather_nd(other_labels, label_pick_inds)

  return sampled_labels

def sample_labels(labels, labels_mask):
  if FLAGS.multi_attr:
    sampled_labels = []

    attrib_mask = tf.tile(
            tf.reshape(tf.constant(FLAGS.attrib_mask), (1, FLAGS.Nattr)),
            (FLAGS.batch_size, 1))
    chosen_attrs = tf.squeeze(tf.multinomial(tf.log(attrib_mask + 1e-10), 1))
    chosen_attrs_oh = tf.one_hot(chosen_attrs, FLAGS.Nattr)

    bin_mask = tf.random_uniform((FLAGS.batch_size, FLAGS.Nattr)) > 0.5
    bin_mask = tf.cast(bin_mask, tf.float32)
    chosen_attrs_oh += bin_mask
    chosen_attrs_oh = chosen_attrs_oh >= 1

    for i in range(FLAGS.Nattr):
      sampled_labels_attr = sample_labels_single_attr(labels[:,i], FLAGS.Nlabels[i])
      sampled_labels_attr = tf.expand_dims(sampled_labels_attr, -1)
      sampled_labels.append(tf.cast(sampled_labels_attr, tf.int64))
    sampled_labels = tf.concat(sampled_labels, axis=1)

    sampled_labels = tf.where(chosen_attrs_oh, sampled_labels, labels)
    return sampled_labels
  else:
    return sample_labels_single_attr(labels, FLAGS.Nlabels)


def interp_z(cnt_enc, sample_cnt_enc):
  if FLAGS.interp_z:
    z = tf.where(
        tf.random_uniform((FLAGS.batch_size, FLAGS.lstm_size), dtype=tf.float32) < FLAGS.interp_ratio,
        cnt_enc, sample_cnt_enc)
  else:
      z = cnt_enc
  return z 

def model(sen_batch, mask_batch, shp, labels_batch, labels_mask, summary_ops):

  (sen_batch, mask_batch, batch_len, seq_len_encoder, seq_len_decoder) = proc(
        sen_batch, mask_batch, shp, labels_batch)

  flip_labels = sample_labels(labels_batch, labels_mask)

  losses = []

  # Word embeddings
  if FLAGS.load_embeddings:
    with open(FLAGS.embeddings_path, 'r') as f:
      word_embs = np.load(f)
    assert word_embs.shape[0] == FLAGS.vocab_size
    word_vectors = tf.get_variable(
        name='rnn/word_embedding', shape=word_embs.shape,
        initializer=tf.constant_initializer(word_embs))
    print('Embeddings loaded')

  else:
    word_vectors = tf.get_variable(
        name='rnn/word_embedding',
        shape=[FLAGS.vocab_size, FLAGS.embedding_size],
        initializer=tf.random_uniform_initializer(
            minval=-FLAGS.uniform_init_scale,
            maxval=FLAGS.uniform_init_scale)
    )

  embs = tf.nn.embedding_lookup(word_vectors, sen_batch)

  # Sample attributes
  labels_oh = one_hot(labels_batch, labels_mask)
  flip_labels_oh = one_hot(flip_labels, labels_mask)

  zero_state = tf.zeros((FLAGS.batch_size, FLAGS.lstm_size))

  cnt_embs = embs

  # Encoder
  with tf.variable_scope('cnt_enc'):
    cell = tf.contrib.rnn.GRUCell(num_units=FLAGS.lstm_size)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell,
        input_keep_prob=FLAGS.dropout_keep_prob_enc)

    cnt_embs_flip = flip_input(cnt_embs, seq_len_encoder)
    cnt_enc_states, cnt_enc = tf.nn.dynamic_rnn(
        cell=cell, dtype=tf.float32, inputs=cnt_embs_flip,
        initial_state=zero_state, sequence_length=seq_len_encoder)

  # Decoder
  with tf.variable_scope('dec'):

    cell = tf.contrib.rnn.GRUCell(num_units=FLAGS.dec_lstm_size)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell,
        input_keep_prob=FLAGS.dropout_keep_prob_dec)

    cnt_enc_flip_mix = mult_int(cnt_enc, flip_labels_oh, reuse=False)

    dec_seq = decode(
        cnt_enc_flip_mix, flip_labels_oh, src_states=cnt_enc_states,
        src_mask=mask_batch, pred_logprobs=True, max_sampling=False)
    (sample, sample_mask, sample_lps, sample_len, sample_states) = dec_seq
    sample_embs = tf.nn.embedding_lookup(word_vectors, sample)

  ########## Second encoder
  with tf.variable_scope('cnt_enc', reuse=True):
    cell = tf.contrib.rnn.GRUCell(num_units=FLAGS.lstm_size)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell,
        input_keep_prob=FLAGS.dropout_keep_prob_enc)

    # Flip input words to Second encoder
    sample_embs_flip = flip_input(sample_embs, sample_len)
    sample_cnt_enc_states, sample_cnt_enc = tf.nn.dynamic_rnn(
        cell=cell, dtype=tf.float32, inputs=sample_embs_flip,
        initial_state=zero_state, sequence_length=sample_len)

  sample_cnt_enc = interp_z(cnt_enc, sample_cnt_enc)

  ############### Second decoder
  # MLE loss
  with tf.variable_scope('dec', reuse=True):
    cell = tf.contrib.rnn.GRUCell(num_units=FLAGS.dec_lstm_size)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell,
        input_keep_prob=FLAGS.dropout_keep_prob_dec)

    enc_rep = sample_cnt_enc
    init_state = mult_int(enc_rep, labels_oh, reuse=True)

    dec_input, initial_state = get_decoder_input_state(
        cnt_embs, init_state, labels_oh, batch_len)

    dec_mle_states, _ = tf.nn.dynamic_rnn(
        cell=cell, dtype=tf.float32, inputs=dec_input,
        initial_state=initial_state)
    rec_loss = classify(dec_mle_states, sen_batch, mask_batch, labels_oh)

    add_loss((rec_loss, 'losses/rec'), losses)

  total_loss = slim.losses.get_total_loss()
  # For summary
  losses.append((total_loss, 'losses/total'))

  gan_loss = {}
  fake = sample_states
  real = dec_mle_states
  masks = [mask_batch, sample_mask]
  with tf.variable_scope('GAN_disc'):
    gan_loss = cgan_classifier(
      sen_batch=[real, fake],
      labels=[labels_oh, flip_labels_oh],
      embedding_size=FLAGS.dec_lstm_size,
      batch_size=FLAGS.batch_size,
      mask_batch=masks,
      mask_zeros=True,
      labels_mask=labels_mask
    )

  for loss_name in ['gen_loss', 'adv_loss', 'adv_acc']:
    summary_ops.append(tf.summary.scalar('GAN/' + loss_name, gan_loss['rf'][loss_name]))

  return losses, total_loss, gan_loss

def model_sample(sen_batch, mask_batch, shp, labels_batch, max_sampling=True,
    reuse=False):

  sen_batch, mask_batch, _, seq_len_encoder, _ = proc(
      sen_batch, mask_batch, shp, labels_batch)
  sen_batch = tf.cast(sen_batch, tf.int64)

  if reuse:
    word_vectors = slim.get_variables_by_name('rnn/word_embedding')[0]
  else:
    word_vectors = tf.get_variable(
        name='rnn/word_embedding',
        shape=[FLAGS.vocab_size, FLAGS.embedding_size])

  embs = tf.nn.embedding_lookup(word_vectors, sen_batch)

  zero_state = tf.zeros((FLAGS.batch_size, FLAGS.lstm_size))

  cnt_embs = embs

  with tf.variable_scope('cnt_enc', reuse=reuse):
    cell = tf.contrib.rnn.GRUCell(num_units=FLAGS.lstm_size)

    cnt_embs_flip = flip_input(cnt_embs, seq_len_encoder)
    cnt_enc_states, cnt_enc = tf.nn.dynamic_rnn(
        cell=cell, dtype=tf.float32, inputs=cnt_embs_flip,
        initial_state=zero_state, sequence_length=seq_len_encoder)

  with tf.variable_scope('dec', reuse=reuse):
    cell = tf.contrib.rnn.GRUCell(num_units=FLAGS.dec_lstm_size)

    all_samples = []

    if FLAGS.multi_attr:

      reuse_flag = False
      attribs_valid = []
      for i in range(FLAGS.Nattr):
        if FLAGS.attrib_mask[i] == 0:
          attribs_valid.append([0])
        else:
          attribs_valid.append(range(FLAGS.Nlabels[i]))
      attribs_valid = itertools.product(*attribs_valid)
      attribs_valid = [list(attribs) for attribs in attribs_valid]

      reuse_flag = False
      for attribs in attribs_valid:
        labels_attr = tf.tile(
                tf.reshape(tf.constant(attribs), (1, FLAGS.Nattr)),
                (FLAGS.batch_size, 1))
        attrib_mask = tf.ones((FLAGS.batch_size, FLAGS.Nattr))
        labels_oh = one_hot(labels_attr, attrib_mask)

        cnt_enc_mix = mult_int(cnt_enc, labels_oh, reuse=reuse_flag)
        samples, _ = decode(
            cnt_enc_mix, labels_oh, src_states=cnt_enc_states,
            src_mask=mask_batch, reuse=reuse_flag)
        all_samples.append(samples)
        reuse_flag = True

    else:

      for l in range(FLAGS.Nlabels):
        labels = l*tf.ones((FLAGS.batch_size,), dtype=tf.int32)
        labels_oh = one_hot(labels)
        cnt_enc_mix = mult_int(cnt_enc, labels_oh, reuse=l > 0 or reuse)
        samples, _ = decode(
            cnt_enc_mix, labels_oh, src_states=cnt_enc_states,
            src_mask=mask_batch, reuse=l > 0 or reuse,
	    max_sampling=max_sampling)
        all_samples.append(samples)

  return all_samples
