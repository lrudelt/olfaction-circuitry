import sys
path = '/data.nst/lucas/MB_learning/'
if path not in sys.path:
    sys.path.insert(1, path)
from classes import *
from plotting import *
from odor_input import *
from utils import *

## Parameters of the run
n_odor = 20
n_ORN = 21
n_neuron = 20
n_train_odor_repeats = 10 ## need 500 to learn fully generally
n_test_odor_repeats = 5 ## each odor presented x times
presentation_length = 1000 #ms
dt = 0.5 #ms
n_train_odors = n_train_odor_repeats*n_odor ## total odors trained
odor_log_interval = 50
folder_name = path + "simulations/220725"
project_name = "save_test"
odor_file = '/data.nst/arana/olfaction_circuitry/odor_data/carlson2008_-2_raw.csv'

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

## Initialize starting weights
starting_feedforward = np.exp(np.maximum(np.random.rand(s.n_neuron, s.n_stim)*0.3-0.2, 0) - 0.99)/2
starting_recurrent = np.zeros((s.n_neuron, s.n_neuron))
diag = np.diag(-np.matmul(starting_feedforward, starting_feedforward.T))
np.fill_diagonal(starting_recurrent, diag)

## Define the odor input files
inputs = get_PN_dynamics(s, num_data=n_train_odors)
test_inputs = get_PN_dynamics(s, num_data=n_test_odor_repeats, all_odors=True)
## in (odor_no, timestep within odor block, num_PNs)

## Init and run net
net = SomaticNet(s)
net.feedforward_weights = starting_feedforward
net.recurrent_weights = starting_recurrent
#in1 = inputs[0, 300, :]
#print(np.matmul(starting_feedforward, in1) + diag/4 + 0.1*net.neuron_biases) ## make sure reasonable chance of some neurons firing
net.run_net(inputs, test_inputs, s)
learning_speed(net, s, odor_file, n_test_odor_repeats, interval=1, plot=True, plateau=5)
#analyze_net(net, s, odor_file, n_test_odor_repeats, save_info=False, plot_info=True, epochs=20)
#analyze_net_valence(net, s, odor_file, n_test_odor_repeats)

random = SomaticNet(s)
s.project_name = s.project_name + "_random"
random.feedforward_weights = randomify_feedforward(net.feedforward_weights)
random.neuron_biases = net.neuron_biases
random.run_net(inputs, test_inputs, s, decoder_only=True)
learning_speed(random, s, odor_file, n_test_odor_repeats, interval=1, plot=True)
#analyze_net(random, s, odor_file, n_test_odor_repeats, save_info=False, plot_info=True, epochs=20)
#analyze_net_valence(random, s, odor_file, n_test_odor_repeats)

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