# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Supervised training for the Grid cell network."""

import matplotlib
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
print(tf.__version__)

matplotlib.use('Agg')
import dataset_reader  # pylint: disable=g-bad-import-order, g-import-not-at-top
import model  # pylint: disable=g-bad-import-order
#import scores  # pylint: disable=g-bad-import-order
import utils  # pylint: disable=g-bad-import-order

FLAGS = {
    'task_dataset_info':'square_room',
    'task_root':'../data',
    'task_env_size':2.2,
    'task_n_pc':[256],
    'task_pc_scale':[0.01],
    'task_n_hdc':[12],
    'task_hdc_concentration':[20.],
    'task_neurons_seed':8341,
    'task_targets_type':'softmax',
    'task_lstm_init_type':'softmax',
    'task_velocity_inputs':True,
    'task_velocity_noise':[0.0,0.0,0.0],
    'model_nh_lstm':128,
    'model_nh_bottleneck':256,
    'model_dropout_rates':[0.5],
    'model_weight_decay':1e-5,
    'model_bottleneck_has_bias':False,
    'model_init_weight_disp':0.0,
    'training_epochs':1000,
    'training_steps_per_epoch':1000,
    'training_minibatch_size':10,
    'training_evaluation_minibatch_size':4000,
    'training_clipping_function':'utils.clip_all_gradients',
    'training_clipping':1e-5,
    'training_optimizer_class':'tf.compat.v1.train.RMSPropOptimizer',
    'training_optimizer_options':'{"learning_rate": 1e-5,"momentum": 0.9}',
    'saver_results_directory':"results",
    'saver_eval_time':2
}


def train():
    """Training loop."""
    # Create the ensembles that provide targets during training
    place_cell_ensembles = utils.get_place_cell_ensembles(
        env_size=FLAGS['task_env_size'],
        neurons_seed=FLAGS['task_neurons_seed'],
        targets_type=FLAGS['task_targets_type'],
        lstm_init_type=FLAGS['task_lstm_init_type'],
        n_pc=FLAGS['task_n_pc'],
        pc_scale=FLAGS['task_pc_scale'])
    head_direction_ensembles = utils.get_head_direction_ensembles(
        neurons_seed=FLAGS['task_neurons_seed'],
        targets_type=FLAGS['task_targets_type'],
        lstm_init_type=FLAGS['task_lstm_init_type'],
        n_hdc=FLAGS['task_n_hdc'],
        hdc_concentration=FLAGS['task_hdc_concentration'])
    target_ensembles = place_cell_ensembles + head_direction_ensembles
    # Store the grid scores
    '''grid_scores = dict()
    grid_scores['btln_60'] = np.zeros((FLAGS['model_nh_bottleneck'],))
    grid_scores['btln_90'] = np.zeros((FLAGS['model_nh_bottleneck'],))
    grid_scores['btln_60_separation'] = np.zeros((FLAGS['model_nh_bottleneck'],))
    grid_scores['btln_90_separation'] = np.zeros((FLAGS['model_nh_bottleneck'],))
    grid_scores['lstm_60'] = np.zeros((FLAGS['model_nh_lstm'],))
    grid_scores['lstm_90'] = np.zeros((FLAGS['model_nh_lstm'],))

    # Create scorer objects
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    masks_parameters = zip(starts, ends.tolist())
    latest_epoch_scorer = scores.GridScorer(20, data_reader.get_coord_range(),
                                            masks_parameters)'''

    #tf.compat.v1.reset_default_graph()

    data_reader = dataset_reader.DataReader(
        FLAGS['task_dataset_info'], root=FLAGS['task_root'], num_threads=4, batch_size=FLAGS['training_minibatch_size'])

    # Model creation
    rnn = model.GridCellNetwork(
        target_ensembles=target_ensembles,
        nh_lstm=FLAGS['model_nh_lstm'],
        nh_bottleneck=FLAGS['model_nh_bottleneck'],
        dropoutrates_bottleneck=np.array(FLAGS['model_dropout_rates']),
        bottleneck_weight_decay=FLAGS['model_weight_decay'],
        bottleneck_has_bias=FLAGS['model_bottleneck_has_bias'],
        init_weight_disp=FLAGS['model_init_weight_disp'])
   
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-5, momentum=0.9, clipvalue=1e-5)
    for epoch in range(1000):
        loss_metric = tf.keras.metrics.Mean()
        for batch in range(1000):
            train_traj = data_reader.read()
            init_pos, init_hd, ego_vel, target_pos, target_hd = train_traj
            input_tensors = []      
            if FLAGS['task_velocity_inputs']:
                vel_noise = tfp.distributions.Normal(0.0, 1.0).sample(
                    sample_shape=tf.shape(ego_vel)) * FLAGS['task_velocity_noise']
                input_tensors = [ego_vel + vel_noise] + input_tensors
            inputs = tf.concat(input_tensors, axis=2)
            initial_conds = utils.encode_initial_conditions(init_pos, init_hd, place_cell_ensembles, head_direction_ensembles)
            ensembles_targets = utils.encode_targets(target_pos, target_hd, place_cell_ensembles, head_direction_ensembles)
            loss, gradients = train_step(inputs, initial_conds, ensembles_targets, rnn)
            back_pass(rnn, optimizer, gradients)
            loss_metric(loss)
        print("epoch {}, loss {}".format(epoch, loss_metric.result()))
        loss_metric.reset_states()

@tf.function
def train_step(inputs, initial_conds, targets, model):
    with tf.GradientTape() as tape:
        outputs, final_state = model(inputs, initial_conds, training=False)
        ensembles_logits, bottleneck, lstm_output = outputs
        loss = loss_func(targets, ensembles_logits)
    gradient = tape.gradient(loss, model.trainable_weights)
    return loss, gradient
@tf.function
def loss_func(ensembles_targets, ensembles_logits):
    pc_loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
        labels=ensembles_targets[0], logits=ensembles_logits[0], name='pc_loss')
    hd_loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
        labels=ensembles_targets[1], logits=ensembles_logits[1], name='hd_loss')
    total_loss = pc_loss + hd_loss
    train_loss = tf.reduce_mean(total_loss, name='train_loss')
    return train_loss
@tf.function
def back_pass(model, optimizer, gradients):
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

def main():
  train()

if __name__ == '__main__':
    main()