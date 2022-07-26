import sys
path = '/data.nst/arana/olfaction_circuitry/'
if path not in sys.path:
    sys.path.insert(1, '/data.nst/arana/olfaction_circuitry/')
from plotting import *
from odor_input import *
from classes import *
from utils import *

## Parameters of the run
n_odor = 15
n_ORN = 20
n_neuron = 50
n_train_odor_repeats = 1000
n_test_odor_repeats = 20 ## each odor presented x times
presentation_length = 1000 #ms
dt = 0.5 #ms
n_train_odors = n_train_odor_repeats*n_odor ## total odors trained
odor_log_interval = 50
folder_name = path + "odor_test/simulations"
project_name = "220720_a"
odor_file = '/data.nst/arana/olfaction_circuitry/odor_test/odor_distinct.csv'

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
s.eta_feedforward = 0.00006/5
s.eta_recurrent = 0.00006/5
s.eta_decoder = 0.0005/5
s.input_selector = odor_flat_linchange
s.odor_file = odor_file
s.input_adaptation = False
s.set('update_interval', presentation_length*n_odor*5) ## for biases - needs to be at least as long as odor presentation time * num odors
s.plasticity = Analytic_Simple()

## Initialize starting weights
starting_feedforward = np.exp(np.maximum(np.random.rand(s.n_neuron, s.n_stim)*0.3-0.2, 0) - 0.99)/2
starting_recurrent = np.zeros((s.n_neuron, s.n_neuron))
diag = np.diag(-np.matmul(starting_feedforward, starting_feedforward.T))
np.fill_diagonal(starting_recurrent, diag)

fig, axs = plt.subplots(1,1)
plot_odor(axs, s, 10)
fig.savefig("test")
#fig, axs = plt.subplots(1,1)
#plot_odor(axs, s, 100)
#fig.savefig("hi")
#print(log.dynamic_log["mean_feedforward_weights"])
#print(max_nonzero(log.dynamic_log["mean_feedforward_weights"]))
#print(log.snapshots[max(log.snapshots.keys())])
#plot_mnist_full("/data.nst/arana/olfaction_circuitry/mnist_test" + "/log")
#plot_mnist_full("/data.nst/arana/olfaction_circuitry/mnist_test" + "/" + "mnist_rate" + "_log")
#plot_odor_full(filename)
