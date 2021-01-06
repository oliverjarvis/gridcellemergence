import collections
import os
import tensorflow as tf
import tree
from os import path

DatasetInfo = collections.namedtuple(
    'DatasetInfo', ['basepath', 'size', 'sequence_length', 'coord_range'])

_DATASETS = dict(
    square_room=DatasetInfo(
        basepath='square_room_100steps_2.2m_1000000',
        size=100,
        sequence_length=100,
        coord_range=((-1.1, 1.1), (-1.1, 1.1))),)

def _get_dataset_files(dateset_info, root):
  """Generates lists of files for a given dataset version."""
  basepath = dateset_info.basepath
  base = os.path.join(root, basepath)
  num_files = dateset_info.size
  template = '{:0%d}-of-{:0%d}.tfrecord' % (4, 4)
  return [
      base + "-" + template.format(i, num_files - 1)
      for i in range(num_files)
  ]

class DataReader(object):
  def __init__(
      self,
      dataset,
      root,
      # Queue params
      num_threads=4,
      capacity=256,
      min_after_dequeue=128,
      seed=None,
      batch_size=0):

    if dataset not in _DATASETS:
      raise ValueError('Unrecognized dataset {} requested. Available datasets '
                       'are {}'.format(dataset, _DATASETS.keys()))

    self._dataset_info = _DATASETS[dataset]
    self._steps = _DATASETS[dataset].sequence_length

    with tf.device('/cpu'):
      file_names = _get_dataset_files(self._dataset_info, root)
      raw_dataset = tf.data.TFRecordDataset(file_names)

      data_feature_description = {
          'init_pos': tf.io.FixedLenFeature(shape=[2], dtype=tf.float32),
          'init_hd': tf.io.FixedLenFeature(shape=[1], dtype=tf.float32),
          'ego_vel': tf.io.FixedLenFeature(shape=[self._dataset_info.sequence_length, 3], dtype=tf.float32),
          'target_pos': tf.io.FixedLenFeature(shape=[self._dataset_info.sequence_length, 2], dtype=tf.float32),
          'target_hd': tf.io.FixedLenFeature(shape=[self._dataset_info.sequence_length, 1], dtype=tf.float32)
      }
      def _parse_data_function(example_proto):
        # Parse the input tf.train.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, data_feature_description)

      parsed_dataset = raw_dataset.map(_parse_data_function).batch(batch_size)
      self.parsed_dataset = tf.compat.v1.data.make_one_shot_iterator(parsed_dataset)
  def read(self):
    traj = self.parsed_dataset.get_next()
    return traj['init_pos'], traj['init_hd'], traj['ego_vel'][:, :self._steps, :], traj['target_pos'][:, :self._steps, :], traj['target_hd'][:, :self._steps, :]

  def get_coord_range(self):
    return self._dataset_info.coord_range