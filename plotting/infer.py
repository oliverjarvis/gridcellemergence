import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
print(tf.__version__)
import pickle
import numpy as np
import pylab as pl
from matplotlib import collections  as mc
import dataset_reader
matplotlib.use('Agg')
import model  # pylint: disable=g-bad-import-order
import scores  # pylint: disable=g-bad-import-order
import utils  # pylint: disable=g-bad-import-order
FLAGS = {
    'task_dataset_info':'square_room',
    'task_root':'data',
    'task_env_size':8,
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
    'training_minibatch_size':32,
    'training_evaluation_minibatch_size':4000,
    'training_clipping_function':'utils.clip_all_gradients',
    'training_clipping':1e-5,
    'training_optimizer_class':'tf.compat.v1.train.RMSPropOptimizer',
    'training_optimizer_options':'{"learning_rate": 1e-5,"momentum": 0.9}',
    'saver_results_directory':"results",
    'saver_eval_time':2
}

#load required files
with open("rat_results.pickle", "rb") as w:
    data = pickle.load(w)
    place_cell_ensembles = data[1]
    head_direction_ensembles = data[2] 

data_reader = dataset_reader.DataReader(
        FLAGS['task_dataset_info'], root=FLAGS['task_root'], num_threads=4, batch_size=FLAGS['training_minibatch_size'])

dataset = data_reader.read()

target_ensembles = place_cell_ensembles + head_direction_ensembles

def create_model(FLAGS, target_ensembles):
    return model.GridCellNetwork(
        target_ensembles=target_ensembles,
        nh_lstm=FLAGS['model_nh_lstm'],
        nh_bottleneck=FLAGS['model_nh_bottleneck'],
        dropoutrates_bottleneck=np.array(FLAGS['model_dropout_rates']),
        bottleneck_weight_decay=FLAGS['model_weight_decay'],
        bottleneck_has_bias=FLAGS['model_bottleneck_has_bias'],
        init_weight_disp=FLAGS['model_init_weight_disp'])

def process_data(data):
    init_pos, init_hd, ego_vel, target_pos, target_hd = data
    init_hd = tf.reshape(init_hd, [-1, 1])
    target_hd = tf.reshape(target_hd, [-1, 100, 1])
    init_pos = tf.cast(init_pos, dtype=tf.float32)
    init_hd = tf.cast(init_hd, dtype=tf.float32)
    ego_vel = tf.cast(ego_vel, dtype=tf.float32)
    target_pos =  tf.cast(target_pos, dtype=tf.float32)
    target_hd = tf.cast(target_hd, dtype=tf.float32)
    return (init_pos, init_hd, ego_vel, target_pos, target_hd)

# Store the grid scores
grid_scores = dict()
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
latest_epoch_scorer = scores.GridScorer(16, ((-8.0, 8.0), (-8.0, 8.0)), masks_parameters)

res_bottleneck = list()
res_lstm_out = list()
res_pos_xy = list()


model = create_model(FLAGS, target_ensembles)

pos_xy = []
bottleneck = []
lstm_output = []
mb_res = dict()
new_res = dict()


#dataset = tf.data.Dataset.from_tensor_slices((data['init_pos'], data['init_hd'], data['ego_vel'], data['target_pos'], data['target_hd'])).batch(32).repeat(10000)

with open("snake_data_1", "rb") as w:
        data = pickle.load(w)

for i in range(len(data['init_pos'])):
    data['init_pos'][i] = np.asarray(data['init_pos'][i]).astype(float) - 8.0
for i in range(len(data['target_pos'])):
    for j in range(len(data['target_pos'][i])):
        data['target_pos'][i][j] = np.asarray(data['target_pos'][i][j]).astype(float) - 8.0

dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(256).batch(10).repeat(200)

first = True

predictions = []
targets = []

for j, train_traj in enumerate(dataset):
    if j == 400:
        break
    train_traj = train_traj['init_pos'], train_traj['init_hd'], train_traj['ego_vel'], train_traj['target_pos'], train_traj['target_hd']
    init_pos, init_hd, ego_vel, target_pos, target_hd = train_traj
    init_hd = tf.reshape(init_hd, [-1, 1])
    target_hd = tf.reshape(target_hd, [10, -1, 1])
    init_pos = tf.cast(init_pos, tf.float32)
    init_hd = tf.cast(init_hd, tf.float32)
    ego_vel = tf.cast(ego_vel, tf.float32)
    target_pos = tf.cast(target_pos, tf.float32)
    target_hd = tf.cast(target_hd, tf.float32)
    inputs = tf.concat(ego_vel, axis=2)
    initial_conds = utils.encode_initial_conditions(init_pos, init_hd, place_cell_ensembles, head_direction_ensembles)
    outputs, final_state = model(inputs, initial_conds, training=False)
    if first:
        model.load_weights("rat_weights.h5")
        first = False
    ensembles_logits, bottleneck, lstm_output = outputs
    for i in range(len(target_pos)):
        mb_res = {"bottleneck":bottleneck[i], "lstm_out":lstm_output[i], "pos_xy":target_pos[i]}
        res_bottleneck.append(np.array(bottleneck[i]))
        res_lstm_out.append(np.array(lstm_output[i]))
        res_pos_xy.append(np.array(target_pos[i]))
    predictions.append(ensembles_logits[0].numpy())
    targets.append(target_pos)

with open("rat_samples", "wb+") as w:
    pickle.dump([predictions, targets], w)

'''
trajectory_predict = ensembles_logits[0].numpy()
trajectory_groundtruth = target_pos
'''

res_bottleneck = np.array(res_bottleneck)
res_lstm_out = np.array(res_lstm_out)
res_pos_xy = np.array(res_pos_xy)
# Store at the end of validation
filename = 'rates_and_sac_latest_hd_' + "lastsnake_32_225_2" + '.pdf'
grid_scores['btln_60'], grid_scores['btln_90'], grid_scores[
    'btln_60_separation'], grid_scores[
        'btln_90_separation'] = utils.get_scores_and_plot(
            latest_epoch_scorer, res_pos_xy, res_bottleneck,
            FLAGS['saver_results_directory'], filename)


def map_trajectory(trajectory, real_points):
    for i in range(0, len(x), 1):
        plt.plot(x[i:i+2], y[i:i+2], 'ro-')
        plt.plot(x[i:i+2], y[i:i+2], 'kx')
    plt.save("sample_trajectory.png")
    plt.show()

map_trajectory(trajectory_predict, trajectory_groundtruth)


#grid cells and select the correct ones
#Some kind of grid-celledness metric
