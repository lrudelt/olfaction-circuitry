#%% 
import sys
path = '/data.nst/lucas/MB_learning'
if path not in sys.path:
    sys.path.insert(1, path)
from classes import *
import plotting
from odor_input import *
from utils import *

#%%
from importlib import reload
plotting = reload(plotting)

#%% 
## Parameters of the run
## Odors maxed, ORN and Neurons as measured
n_odor = 10
n_ORN = 21
n_neuron = 54
n_train_odor_repeats = 50 ## need 500 to learn fully generally
n_test_odor_repeats = 20 ## each odor presented x times
presentation_length = 500 #ms
dt = 0.5 #ms
n_train_odors = n_train_odor_repeats*n_odor ## total odors trained
odor_log_interval = 200 # in number of odors
folder_name = path + "simulations"
project_name = "samuel2019"
odor_file = path + 'odor_data/samuel2019_-4_raw_rounded.csv'
specificity = n_neuron/n_odor
homeostatic_bias = False # Do NOT adapt the bias to achieve a fixed firing rate per neuron
# baseline_firing_rate = specificity/(n_odor*s.kernel_tau)
baseline_firing_rate = 1/1000. # kHz (rate per millisecond) spontaneous firing
sigma = np.sqrt(0.1)

## Define the settings dictionary from the standard one
s = Settings()
s.set('presentation_length', presentation_length) ## ms here since default dt is 1
s.set('fade_fraction', 0.2)
s.set('input_tau_inc', 50)
s.set("initial_sigma", sigma) 
s.set('rho', baseline_firing_rate) # in case of analytic threshold, this sets the spontaneous firing rate due to the bias
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
s.eta_bias = 0.0005
s.eta_decoder = 0.00005
s.input_noise = True # Attempt to make firing more decoupled by introducing noise
## Since presentation length increased 10-fold, need to also change learning rates to decrease 10+-fold
s.input_selector = get_odor_rate
s.odor_file = odor_file
s.MBON_settings_dict = {"c": 10, "learning_rate": 0.000001}
# s.eta_bias = 0 # Unset plasticity TODO: build this in properly
s.set('update_interval', s.presentation_length*n_odor) ## for biases - needs to be at least as long as odor presentation time * num odors
s.plasticity = Analytic_Simple()
s.description = "High eta_bias, four runs for optimal, random with optimal F, measured, and random with measured F"

## Define the odor input files
inputs = get_PN_dynamics(s, num_data=n_train_odors)
test_inputs = get_PN_dynamics(s, num_data=n_test_odor_repeats, all_odors=True)
## in (odor_no, timestep within odor block, num_PNs)

## Initialize starting weights
# starting_feedforward = np.exp(np.maximum(np.random.rand(s.n_neuron, s.n_stim)*0.3-0.2, 0) - 0.99)/2
# diag = np.diag(-np.matmul(starting_feedforward, starting_feedforward.T))

## Init and run net
if False: 
    s.project_name = project_name + "_opt"
    net = SomaticNet(s)
    net.feedforward_weights = starting_feedforward
    np.fill_diagonal(net.recurrent_weights, diag)
    #in1 = inputs[0, 300, :]
    #print(np.matmul(starting_feedforward, in1) + diag/4 + 0.1*net.neuron_biases) ## make sure reasonable chance of some neurons firing
    net.run_net(inputs, test_inputs, s)
    learning_speed(net, s, odor_file, n_test_odor_repeats, interval=1, plot=True, plateau=10)

if False: 
    random = SomaticNet(s)
    s.project_name = project_name + "_opt_rand"
    random.feedforward_weights = randomify_feedforward(net.feedforward_weights)
    #random.neuron_biases = net.neuron_biases
    random.run_net(inputs, test_inputs, s, decoder_only=True)
    learning_speed(random, s, odor_file, n_test_odor_repeats, interval=1, plot=True, plateau=10)

#%%
meas = SomaticNet(s)
s.project_name = project_name + "_meas"
meas_file = path + "connectivity_data/PN_KC_Meas_left.csv"
meas_feed = read_meas_connectivity(meas_file)/10
meas.feedforward_weights = meas_feed
#meas.neuron_biases = net.neuron_biases
diag = np.diag(-np.matmul(meas_feed, meas_feed.T))
np.fill_diagonal(meas.recurrent_weights, diag)
# in1 = inputs[0, 300, :]
# print(np.matmul(meas_feed, in1) + diag/4 + 0.1*meas.neuron_biases) ## make sure reasonable chance of some neurons firing
meas.run_net(inputs, test_inputs, s, decoder_only=True, homeostatic_bias=homeostatic_bias)
# learning_speed(meas, s, odor_file, n_test_odor_repeats, interval=1, plot=True, plateau=10)

if False: 
    meas_rand = SomaticNet(s)
    s.project_name = project_name + "_meas_rand"
    meas_rand.feedforward_weights = randomify_feedforward(meas.feedforward_weights)
    meas_rand.run_net(inputs, test_inputs, s, decoder_only=True)
    learning_speed(meas_rand, s, odor_file, n_test_odor_repeats, interval=1, plot=True, plateau=10)


# %% Load log
log_file = "../simulations/samuel2019_meas_log"
with gzip.open(log_file, 'r') as f:
    log = pickle.load(f)
assert isinstance(log, Log)
final_time = max(log.snapshots.keys())
if final_time > log.dynamic_log["t"][-1]:
    dyn_log_idx = len(log.dynamic_log["t"])
else:
    dyn_log_idx = np.where(log.dynamic_log["t"] == final_time)[0][0]
timetrace = log.dynamic_log["t"][:dyn_log_idx]
decoder_loss = log.dynamic_log["test_decoder_loss"][:dyn_log_idx]
firing_rate = log.dynamic_log["avg_firing_rate"][:dyn_log_idx]
final_snap = log.snapshots[max(log.snapshots.keys())]
F, D, W = final_snap["feedforward_weights"], final_snap["decoder"], final_snap["recurrent_weights"]
input, recon = final_snap["inputs"], final_snap["reconstruction_means"]
spikes = final_snap["spikes"]
try:
    s = final_snap["s"]
except:
    s = Settings()
    s.presentation_length = 1000
    s.snapshot_log_interval = 1
print("Time:", max(log.snapshots.keys()), log.dynamic_log["t"][dyn_log_idx-1])


# %% Produce plots
fig = plt.figure(figsize=(25, 25))
plt.rcParams['figure.facecolor'] = 'white'
#plt.tight_layout()
grid = gridspec.GridSpec(10, 8, hspace=2.5, wspace=1)
dec = plt.subplot(grid[0:2, :])
rat = plt.subplot(grid[2:4, :])
inp = plt.subplot(grid[4:5 , 0:4])
rec = plt.subplot(grid[4:5 , 4:8])
odo = plt.subplot(grid[5    , :])
ras = plt.subplot(grid[6:8: , :])
Fax = plt.subplot(grid[8:10 , 0:4])
Dax = plt.subplot(grid[8:10 , 4:8])
# sp1 = plt.subplot(grid[8:10, 0:2])
# sp2 = plt.subplot(grid[8:10, 2:4])
# acc = plt.subplot(grid[8:10, 4:6])
# con = plt.subplot(grid[8:10, 6:7])
# par = plt.subplot(grid[8:10, 7])


plot_dynamic_var(dec, timetrace, decoder_loss, "Decoder Loss")
plot_dynamic_var(rat, timetrace, firing_rate, "Mean Firing Rate (spikes/neuron/ms)")
plot_matrix(Fax, F.transpose())
plot_matrix(Dax, D)
#file_array = np.loadtxt(filename, dtype=int, skiprows=1, usecols=range(1, 25), delimiter=",")
#odor_matrix = file_array[:s.n_odors, :s.n_ORN]
plot_matrix(inp, input[:10, s.input_tau_inc, :])
plot_matrix(rec, recon[:10])
# if rates is not None:
#     plot_sparsity_fractions(sp1, sp2, rates, s.presentation_length, s.fade_fraction)
plot_raster(ras, spikes, s, nneuron=s.n_neuron, nodors=5)
# plot_odor(odo, s, 5)
# plot_confusion_matrix(con, confusion, "Confusion")
# plot_training_metric(acc, hist)
#plot_pars(par, s)
plt.savefig("%s/%s" % (s.folder_name, s.project_name))
plt.close()

# %%
