import sys
path = '/data.nst/lucas/MB_learning/'
if path not in sys.path:
    sys.path.insert(1, path)
from classes import *
from plotting import *
from odor_input import *
from utils import *

## Parameters of the run
## Odors maxed, ORN and Neurons as measured
n_odor = 34
n_ORN = 21
n_neuron = 56
n_train_odor_repeats = 500 ## need 500 to learn fully generally
n_test_odor_repeats = 50 ## each odor presented x times
presentation_length = 1000 #ms
dt = 0.5 #ms
n_train_odors = n_train_odor_repeats*n_odor ## total odors trained
odor_log_interval = 50
folder_name = path + "simulations/220725"
project_name = "meas_comp_samuel2019"
odor_file = path + 'odor_data/samuel2019_-4_raw.csv'

## Define the settings dictionary from the standard one
s = Settings()
s.set('presentation_length', presentation_length) ## ms here since default dt is 1
s.set('fade_fraction', 0.2)
s.set('input_tau_inc', 50)
s.set('rho', 0.001)
s.set('dt', dt)
s.set('n_odors', n_odor)
s.set('n_ORN', n_ORN)
s.set('n_stim', n_ORN)
s.set('n_neuron', n_neuron)
s.set('folder_name', "'%s'" % folder_name)
s.set('project_name', "'%s'" % project_name)
s.set('dynamic_log_interval', s.presentation_length*odor_log_interval)
s.set('dynamic_sample_interval', 5)
s.set('dynamic_log_test_set', n_odor*2)
s.set('n_train_repeat', n_train_odor_repeats)
s.set('n_test_repeat', n_test_odor_repeats)
## Since presentation length increased 10-fold, need to also change learning rates to decrease 10+-fold
s.input_selector = get_odor_rate
s.odor_file = odor_file
s.MBON_settings_dict = {"c": 10, "learning_rate": 0.000001}
s.set('update_interval', presentation_length*n_odor*5) ## for biases - needs to be at least as long as odor presentation time * num odors
s.plasticity = Analytic_Simple()

log_file = folder_name+"/"+project_name + "_log"
with gzip.open(log_file, 'r') as f:
    log = pickle.load(f)
assert isinstance(log, Log)
final_time = max(log.snapshots.keys())
final_snap = log.snapshots[final_time]
s = final_snap["s"]
print("Odors Trained on:", int(final_time/s.presentation_length))

net = SomaticNet(s)
net.feedforward_weights = final_snap["feedforward_weights"]
net.recurrent_weights = final_snap["recurrent_weights"]
net.neuron_biases = final_snap["biases"]
net.decoder_matrix = final_snap["decoder"]

meas_file = path + "connectivity_data/PN_KC_Meas.csv"

meas = read_meas_connectivity(meas_file, s)
meas_input_corr = compute_PN_input_correlations(meas, s)
print("Measured input correlations: ", meas_input_corr)

meas_io_corr = compute_input_output_correlations(meas, s)
print("Measured input output correlations: ", meas_io_corr)

opt_map, opt_mse = map_sim_to_meas_connectivity(net, meas, s)
print("Optimum mapping: ", opt_map)
print("Optimum MSE: ", opt_mse, np.mean(opt_mse))
## Plot MSE here
opt_input_corr = compute_PN_input_correlations(net.feedforward_weights, s)
print("Optimum input correlations: ", opt_input_corr)

opt_io_corr = compute_input_output_correlations(net.feedforward_weights, s)
print("Optimum input output correlations: ", opt_io_corr)


random = SomaticNet(s)
log_file = folder_name+"/"+project_name + "_random" + "_log"
with gzip.open(log_file, 'r') as f:
    log = pickle.load(f)
assert isinstance(log, Log)
final_time = max(log.snapshots.keys())
final_snap = log.snapshots[final_time]
s = final_snap["s"]
print("Odors Trained on:", int(final_time/s.presentation_length))
random.feedforward_weights = final_snap["feedforward_weights"]
random.recurrent_weights = final_snap["recurrent_weights"]
random.neuron_biases = final_snap["biases"]
random.decoder_matrix = final_snap["decoder"]


ran_map, ran_mse = map_sim_to_meas_connectivity(random, meas, s)
print("Random mapping: ", ran_map)
print("Random MSE: ", ran_mse, np.mean(ran_mse))

ran_input_corr = compute_PN_input_correlations(random.feedforward_weights, s)
print("Random input correlations: ", ran_input_corr)

ran_io_corr = compute_input_output_correlations(random.feedforward_weights, s)
print("Random input output correlations: ", ran_io_corr)
## One idea for classification: reward function from odor to [-1, 1]; use NN with one output node to classify this; single metric
## But original idea is have a 110 layer output function with softmax for interpretation; densify these connections and learn
## Could do sparsity as you can do L1 adaptation as db = r_1 - 0.1;
## TODO:
## Random network by doing probabilistic version of nature paper; could also do binary and see what happens
## 5. Pruning: Do the derivation for Laplace multiplier sparsity constraint leading to different weight decay trick
## 6. Finish compensatory variability paper and check if we see these correlations or not in our learned network
## 8. Andre's data as starting feedforward, do other metrics
## Do param search on rho to find best one?
## Try with spikes added instead of plain rates
## Try on original dataset

## 50 neurons, 50 + 20 odors
## Implement plot where train classifier with different amounts of training data (same epochs to converge) and see learning rate

## Parameters to add to plot:
## 1. Input data: dataset, n_odor, n_ORN
## 2. Network: n_neuron, rho, train size, test size, plasticity, learning rates
## 3. Input dynamics: shape of profile, presentation length, stim_on_frac

## Compute MSE between overlap of model and experiment neuron connectivity for random and model:
# - Look at each optimized KC and find the best match in measured. Then, compute MSE totally and check against random.
## Correlations between PNs inputting to singular KC to average correlation weight for random, model and data; we'd expect this is high
## Run the network on the measured feedforward weights instead and see what happens for classification

## To do thresholding, can set relu threshold and see when the reconstruction fails and find that threshold; then fix F and train 
## Then sparse connectivity can do find PNs with nonzero weights, and find average (weighted by product of weights?) of pairwise correlations of firing rates over different odors
## Newer paper has non-random PN-KC connectivity - see if you can reproduce that result

## Our network better works with more vs less frequent odors
## But generalizes worse with new odors



## Concentration to PN noise data: 4 data points from Olsen 2010 & also 19 stimuli for 3 concentrations in Hallem & Carlson
## Fig 3 from Hallem & Carlson 2006 for 4 ORNs shows no obvious pattern in concentration-ORN - diff for each odor.
## Fig 4, Table S2 has 24 ORNs for 10 pure, 9 fruit odors - no clear pattern here too, so heuristic seems wise to do the
## std dev of inverse tuning curve as concentration proxy FILL PAPER HERE
## At low concentrations of 1e-8, generally only 1 or 2 odors responded even for the broadly tuned receptors