import argparse
import logging
import os
import sys
from pprint import pprint

import matplotlib.pyplot as plt
# %pylab inline
import numpy as np
import tensorflow as tf
import yaml
from astropy.table import Table
from sklearn.metrics import roc_curve

sys.path.insert(0, '../../')
from cosmoml.estimators import RNNClassifier


# ------------------------------
#         CONFIG
# ------------------------------
parser = argparse.ArgumentParser(description="Train a rnn classifier")
parser.add_argument('-cfg', '--config', type=str, nargs='?',
                    help='configuration file path',
                    default='./config/simple_config.yaml')
                    # default='./config/bi_attention_config.yaml')
args = parser.parse_args()

with open(args.config, 'r') as f:
  config = yaml.load(f)

print('=' * 100)
print("CONFIGURATIONS")
pprint(config)
print('=' * 100)

# ------------------------------
#       LOGGING HANDLING
# ------------------------------
# https://stackoverflow.com/questions/40559667/how-to-redirect-tensorflow-logging-to-a-file
# get TF logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = \
  logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logging.getLogger().setLevel(logging.DEBUG)

# create file handler which logs even debug messages
fh = logging.FileHandler('log_{}.txt'.format(config['generic']['name']))
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

# apply handler
logger.addHandler(fh)

# log config
logger.log(logging.DEBUG, config)



# ------------------------------
#         DATA LOADING
# ------------------------------
# Loading dataset
data_table = Table.read('ligthcurve_data.fits.gz')

# Splitting training and testing data
randomize_inds = range(len(data_table))
randomize_inds = np.random.permutation(randomize_inds)
randomized_inds_train = randomize_inds[0:70000]
randomized_inds_test = randomize_inds[70000:]


# ------------------------------
#            MODEL
# ------------------------------
# Define input function for training
def input_fn_train():
  def mapping_function(x):
    def extract_batch(inds):
      inds = randomized_inds_train[inds]
      return data_table['coadd_label'][inds].astype('float32'), \
             np.clip(data_table['obs_len'][inds], 0, 89).astype('int32'), \
             np.clip(data_table['time_series'][inds].astype('float32'), -10, 10)

    a, b, c = tf.py_func(extract_batch, [x], [tf.float32, tf.int32, tf.float32])
    a.set_shape([None])
    b.set_shape([None])
    c.set_shape([None, 90, 12])
    return a, b, c

  dataset = tf.data.Dataset.range(len(randomized_inds_train))
  dataset = dataset.repeat().shuffle(50000).batch(config['train']['batch_size'])
  dataset = dataset.map(mapping_function)
  iterator = dataset.make_one_shot_iterator()
  label, length, ts = iterator.get_next()
  return {'length': length, 'ts': ts}, label


# Define input function for testing on the training set
def input_fn_train_test():
  def mapping_function(x):
    def extract_batch(inds):
      inds = randomized_inds_train[inds]
      return data_table['coadd_label'][inds].astype('float32'), \
             np.clip(data_table['obs_len'][inds], 0, 89).astype('int32'), \
             np.clip(data_table['time_series'][inds].astype('float32'), -10, 10)

    a, b, c = tf.py_func(extract_batch, [x], [tf.float32, tf.int32, tf.float32])
    a.set_shape([None])
    b.set_shape([None])
    c.set_shape([None, 90, 12])
    return a, b, c

  dataset = tf.data.Dataset.range(len(randomized_inds_train))
  dataset = dataset.batch(config['train']['batch_size'])
  dataset = dataset.map(mapping_function)
  iterator = dataset.make_one_shot_iterator()
  label, length, ts = iterator.get_next()
  return {'length': length, 'ts': ts}, label


# Define input function for testing on the testing set
def input_fn_test():
  def mapping_function(x):
    def extract_batch(inds):
      inds = randomized_inds_test[inds]
      return data_table['coadd_label'][inds].astype('float32'), \
             np.clip(data_table['obs_len'][inds], 0, 89).astype('int32'), \
             np.clip(data_table['time_series'][inds].astype('float32'), -10, 10)

    a, b, c = tf.py_func(extract_batch, [x], [tf.float32, tf.int32, tf.float32])
    a.set_shape([None])
    b.set_shape([None])
    c.set_shape([None, 90, 12])
    return a, b, c

  dataset = tf.data.Dataset.range(len(randomized_inds_test))
  dataset = dataset.batch(config['train']['batch_size'])
  dataset = dataset.map(mapping_function)
  iterator = dataset.make_one_shot_iterator()
  label, length, ts = iterator.get_next()
  return {'length': length, 'ts': ts}, label


# ------------------------------
#          INSTANTIATION
# ------------------------------
# Create the model
model = RNNClassifier(hidden_units=config['model']['hidden_units'],
                      in_dropout=config['model']['in_dropout'],
                      out_dropout=config['model']['out_dropout'],
                      is_bidirectional=config['model']['is_bidirectional'],
                      is_attended=config['model']['is_attended'],
                      model_dir=config['train']['model_dir'])


# ------------------------------
#            TRAINING
# ------------------------------
model.train(input_fn_train,
            steps=config['train']['steps'])

# ------------------------------
#            TESTING
# ------------------------------
# Apply model to testing set
test_prob = [p['prob'] for p in model.predict(input_fn_test)]
table_test = data_table[randomized_inds_test]
table_test['p'] = test_prob

# Apply model to training set
train_prob = [p['prob'] for p in model.predict(input_fn_train_test)]
table_train = data_table[randomized_inds_train]
table_train['p'] = train_prob

# ------------------------------
#            REPORTING
# ------------------------------
# Compute ROC curves
fpr1, tpr1, thr1 = roc_curve(table_train['coadd_label'], table_train['p'])
fpr2, tpr2, thr2 = roc_curve(table_test['coadd_label'], table_test['p'])

plt.plot(fpr1, tpr1, label='training set')
plt.plot(fpr2, tpr2, label='testing set')
plt.grid('on')
plt.xscale('log')
plt.legend()
plt.axvline(0.0043)
plt.savefig(os.path.join(config['train']['model_dir'], "ROC_Curve.jpg"))

# Computes colors
table_test['ug'] = table_test['coadd_psfMag_u'] - table_test['coadd_psfMag_g']
table_test['gr'] = table_test['coadd_psfMag_g'] - table_test['coadd_psfMag_r']
table_test['ri'] = table_test['coadd_psfMag_r'] - table_test['coadd_psfMag_i']
table_test['iz'] = table_test['coadd_psfMag_i'] - table_test['coadd_psfMag_z']

threshold = 0.90
# Splits the testing set
FP = (table_test['p'] > threshold) * (table_test['coadd_label'] == 0)
FN = (table_test['p'] < threshold) * (table_test['coadd_label'] == 1)
P = (table_test['coadd_label'] == 1)
N = (table_test['coadd_label'] == 0)

t_fp = table_test[FP]
t_fn = table_test[FN]
t_p = table_test[P]
t_n = table_test[N]

print(len(table_test[FP]))
print(len(table_test[FN]))

# Plot light curve for a false positive
plt.plot(t_fp['time_series'][0, :, 0], t_fp['time_series'][0, :, 1], '+')
plt.plot(t_fp['time_series'][0, :, 0], t_fp['time_series'][0, :, 2], '+')
plt.plot(t_fp['time_series'][0, :, 0], t_fp['time_series'][0, :, 3], '+')
plt.plot(t_fp['time_series'][0, :, 0], t_fp['time_series'][0, :, 4], '+')
plt.xlim(0, 1)
plt.savefig(os.path.join(config['train']['model_dir'], "false_positive.jpg"))

# False positive in color space
plt.figure(figsize=(10, 5))
plt.title('False positive in color space')
plt.subplot(121)
plt.hist2d(table_test['ug'], table_test['gr'], 100, cmap='gist_stern')
plt.scatter(t_fp['ug'], t_fp['gr'], alpha=0.5, c='y', marker='+')
plt.xlabel('u - g')
plt.ylabel('g - r')

plt.subplot(122)
plt.hist2d(table_test['gr'], table_test['ri'], 100, cmap='gist_stern')
plt.scatter(t_fp['gr'], t_fp['ri'], alpha=0.5, c='y', marker='+')
plt.xlabel('g - r')
plt.ylabel('r - i')

plt.suptitle('Variability only false positives', fontsize=16)
plt.savefig(os.path.join(config['train']['model_dir'], "false_negative.jpg"))

# Looks at the distribution of these in terms of coadded imag
plt.hist(t_n['coadd_psfMag_i'], 100)
plt.hist(t_p['coadd_psfMag_i'], 100)
plt.hist(t_fn['coadd_psfMag_i'], 100)
plt.hist(t_fp['coadd_psfMag_i'], 100)
plt.savefig(os.path.join(config['train']['model_dir'],
                         "coadded_distribution.jpg"))
