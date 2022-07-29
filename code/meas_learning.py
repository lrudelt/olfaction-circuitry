import sys
path = '/data.nst/arana/olfaction_circuitry/'
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
folder_name = path + "simulations/220726"
project_name = "samuel2019"
odor_file = path + 'odor_data/samuel2019_-4_raw_rounded.csv'
specificity = n_neuron/n_odor
## Define the settings dictionary from the standard one
s = Settings()
s.set('presentation_length', presentation_length) ## ms here since default dt is 1
s.set('fade_fraction', 0.2)
s.set('input_tau_inc', 50)
firing_rate = specificity/(n_odor*s.kernel_tau)
s.set('rho', firing_rate)
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
s.eta_bias = 0.001
s.set('update_interval', s.presentation_length*n_odor) ## for biases - needs to be at least as long as odor presentation time * num odors
s.plasticity = Analytic_Simple()
s.description = "High eta_bias, four runs for optimal, random with optimal F, measured, and random with measured F"

## Define the odor input files
inputs = get_PN_dynamics(s, num_data=n_train_odors)
test_inputs = get_PN_dynamics(s, num_data=n_test_odor_repeats, all_odors=True)
## in (odor_no, timestep within odor block, num_PNs)

## Initialize starting weights
starting_feedforward = np.exp(np.maximum(np.random.rand(s.n_neuron, s.n_stim)*0.3-0.2, 0) - 0.99)/2
diag = np.diag(-np.matmul(starting_feedforward, starting_feedforward.T))

## Init and run net
s.project_name = project_name + "_opt"
net = SomaticNet(s)
net.feedforward_weights = starting_feedforward
np.fill_diagonal(net.recurrent_weights, diag)
#in1 = inputs[0, 300, :]
#print(np.matmul(starting_feedforward, in1) + diag/4 + 0.1*net.neuron_biases) ## make sure reasonable chance of some neurons firing
net.run_net(inputs, test_inputs, s)
learning_speed(net, s, odor_file, n_test_odor_repeats, interval=1, plot=True, plateau=10)

random = SomaticNet(s)
s.project_name = project_name + "_opt_rand"
random.feedforward_weights = randomify_feedforward(net.feedforward_weights)
#random.neuron_biases = net.neuron_biases
random.run_net(inputs, test_inputs, s, decoder_only=True)
learning_speed(random, s, odor_file, n_test_odor_repeats, interval=1, plot=True, plateau=10)


meas = SomaticNet(s)
s.project_name = project_name + "_meas"
s.eta_bias = 0.0005
meas_file = path + "connectivity_data/PN_KC_Meas.csv"
meas_feed = read_meas_connectivity(meas_file)/10
meas.feedforward_weights = meas_feed
#meas.neuron_biases = net.neuron_biases
diag = np.diag(-np.matmul(meas_feed, meas_feed.T))
np.fill_diagonal(meas.recurrent_weights, diag)
# in1 = inputs[0, 300, :]
# print(np.matmul(meas_feed, in1) + diag/4 + 0.1*meas.neuron_biases) ## make sure reasonable chance of some neurons firing
meas.run_net(inputs, test_inputs, s, decoder_only=True)
learning_speed(meas, s, odor_file, n_test_odor_repeats, interval=1, plot=True, plateau=10)

meas_rand = SomaticNet(s)
s.project_name = project_name + "_meas_rand"
meas_rand.feedforward_weights = randomify_feedforward(meas.feedforward_weights)
meas_rand.run_net(inputs, test_inputs, s, decoder_only=True)
learning_speed(meas_rand, s, odor_file, n_test_odor_repeats, interval=1, plot=True, plateau=10)