"""Training the model
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim
from model import model
from data import data

# Data
tf.flags.DEFINE_string('file_pattern', 'yelp', 'Dataset')
tf.flags.DEFINE_boolean('crop_batches', False, 'Dataset')
tf.flags.DEFINE_integer('num_examples', 500000, 'Number of data instances')

# Basic hparams
tf.flags.DEFINE_float('Nepochs', 20, 'Number of epochs')
tf.flags.DEFINE_integer('batch_size', 10, 'batch size')
tf.flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate')
tf.flags.DEFINE_float('clip_grad_norm', 5, 'gradient clipping norm')
tf.flags.DEFINE_float('uniform_init_scale', 0.1, 'weights init scale')

# Model parmas
tf.flags.DEFINE_integer('vocab_size', 10000, 'vocab size')
tf.flags.DEFINE_integer('embedding_size', 300, 'word size')
tf.flags.DEFINE_integer('lstm_size', 500, 'lstm size')
tf.flags.DEFINE_integer('dec_lstm_size', 500, 'lstm size')
tf.flags.DEFINE_integer('input_size', 30, 'length of sentence')

# Dropout params
tf.flags.DEFINE_float('dropout_keep_prob_enc', 1.0, 'Enc dropout')
tf.flags.DEFINE_float('dropout_keep_prob_dec', 1.0, 'Dec dropout')

# Checkpointing, summaries
tf.flags.DEFINE_string('checkpoint_dir', 'ckpt', 'dir')
tf.flags.DEFINE_integer('ckpt_freq', 300, 'Model eval frequency')
tf.flags.DEFINE_integer('stdout_freq', 120, 'Stdout freq')
tf.flags.DEFINE_integer('summary_freq', 60, 'Summary freq')
tf.flags.DEFINE_integer("max_ckpts", 5, "Max number of ckpts to keep")

# Loading pre-trained models
tf.flags.DEFINE_boolean('load_mdl', False, 'load mdl')
tf.flags.DEFINE_boolean('load_embeddings', False, 'Load word embeddings')
tf.flags.DEFINE_string('embeddings_path', 'vocab.npy', 'embs file')

# GAN parameters
tf.flags.DEFINE_float('adv_weight', 1.0, 'Adversary weight')
tf.flags.DEFINE_float('gen_weight', 1.0, 'Generator weight')
tf.flags.DEFINE_boolean('cgan', False, 'Joint classifier/GAN')

# Other
tf.flags.DEFINE_string('mode', 'train', 'Mode')
tf.flags.DEFINE_integer('decode_max_steps', 30, 'max decode steps')
tf.flags.DEFINE_boolean('flip_input', False, 'Flip input order')
tf.flags.DEFINE_boolean('interp_z', False, 'z interpolation')
tf.flags.DEFINE_float('interp_ratio', 0.5, 'Attr confidence')
tf.flags.DEFINE_boolean('custom_input', False, 'Custom input')

# Attributes
tf.flags.DEFINE_boolean('multi_attr', False, 'Multiple attributes')
tf.flags.DEFINE_integer('Nattr', 1, 'Number of attributes')
tf.flags.DEFINE_string('Nlabels', "2", 'Number of attribute labels')
tf.flags.DEFINE_string('attrib_mask', '1', 'Multiclass labels')
tf.flags.DEFINE_boolean('beam_search', False, 'Beam search')

FLAGS = tf.flags.FLAGS

def print_vars(var_list, name):
  print(name + ': ', [var.name for var in var_list])

def train():

  if FLAGS.multi_attr:
    FLAGS.attrib_mask = map(float, FLAGS.attrib_mask.split(','))
    FLAGS.Nlabels = map(int, FLAGS.Nlabels.split(','))
  else:
    FLAGS.Nlabels = int(FLAGS.Nlabels)

  with tf.Graph().as_default():

    with tf.device(tf.train.replica_device_setter(0)):

      summary_ops = []

      sen_batch, mask_batch, shp, labels_batch, labels_mask = data(FLAGS.file_pattern,
          FLAGS.batch_size)
      losses, total_loss, gan_loss = model(sen_batch, mask_batch, shp, labels_batch,
	  labels_mask, summary_ops)

      for loss in losses:
        summary_ops.append(tf.summary.scalar(loss[1], loss[0]))

      optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
      trainable_vars = tf.trainable_variables()

      init_ops = []
      dict_items = []

      if FLAGS.cgan:

        adv_cls_vars = slim.get_variables(scope='GAN_disc')

        adv_cls_vars = list(set(adv_cls_vars) & set(trainable_vars))

        trainable_vars = list(set(trainable_vars) - set(adv_cls_vars))

        print_vars(adv_cls_vars, 'adv cls vars')
        print_vars(trainable_vars, 'trainable vars')

        adv_losses = []
        gen_losses = [total_loss]

        for name, loss in gan_loss.items():

          adv_perf_good = tf.greater(loss['adv_acc'], 0.75)
          adv_perf_ntgd = tf.less(loss['adv_acc'], 0.98) 

          adv_loss = loss['adv_loss']

          adv_loss *= FLAGS.adv_weight

          adv_loss_cond = tf.cond(adv_perf_ntgd, lambda: adv_loss, lambda: 0.0)
          gen_loss_cond = tf.cond(adv_perf_good, lambda: FLAGS.gen_weight*loss['gen_loss'], lambda: 0.0)

          adv_losses.append(adv_loss_cond)
          gen_losses.append(gen_loss_cond)

        train_op_adv = slim.learning.create_train_op(
            tf.add_n(adv_losses),
            optimizer,
            clip_gradient_norm=FLAGS.clip_grad_norm,
            variables_to_train=adv_cls_vars,
        )
        train_op_gen = slim.learning.create_train_op(
            tf.add_n(gen_losses),
            optimizer,
            clip_gradient_norm=FLAGS.clip_grad_norm,
            variables_to_train=trainable_vars,
        )
        train_op = tf.group(train_op_gen, train_op_adv)
     
      else:

        print('Trainable vars: ', [var.name for var in trainable_vars])
        train_op = slim.learning.create_train_op(
            total_loss,
            optimizer,
            clip_gradient_norm=FLAGS.clip_grad_norm,
            variables_to_train=trainable_vars
        )

      init_op = tf.group(*init_ops)

      init_dict = dict(dict_items)

      def InitAssignFn(sess):
        sess.run(init_op, init_dict)

      nsteps = int(FLAGS.Nepochs * FLAGS.num_examples / FLAGS.batch_size)

      load_params = FLAGS.load_mdl

      if FLAGS.max_ckpts != 5:
        saver = tf.train.Saver(max_to_keep=FLAGS.max_ckpts)
      else:
        saver = tf.train.Saver()

    loss = slim.learning.train(
        train_op,
        master='',
        is_chief=True,
        logdir=FLAGS.checkpoint_dir,
        number_of_steps=nsteps,
        save_summaries_secs=FLAGS.summary_freq,
        log_every_n_steps=FLAGS.stdout_freq,
        save_interval_secs=FLAGS.ckpt_freq,
        summary_op=tf.summary.merge(summary_ops),
        init_fn=InitAssignFn if load_params else None,
        saver=saver
    )

def main(argv=()):
  del argv  # Unused.

  train()

main()
