import sys
sys.path.insert(1, '/data.nst/arana/olfaction_circuitry/')
from classes import *
from plotting import *
from utils import *

## Define the settings dictionary from the standard one
s = Settings()
n_train_patterns = 1000 #50000
n_pattern_snap = 1000
n_test_patterns = 100#000
nNumbers = 3
dt = 0.1
folder = '/data.nst/arana/olfaction_circuitry/mnist_test/simulations'
project = "mnist_0713_update"
s.set('dt', dt)
## Presentation length is set to 100 ms
s.set('n_stim', 16*16)
s.set('n_neuron', 9)
s.set('folder_name', "'%s'" % folder)
s.set('project_name', "'%s'" % project)
s.set('dynamic_log_interval', s.presentation_length)
s.set('dynamic_sample_interval', 5)
s.set('snapshot_log_interval', 1)
s.set('update_interval', 5000)
s.set('snapshot_times', [x for x in range(0, s.presentation_length*n_train_patterns, s.presentation_length*n_pattern_snap)])
s.plasticity = Analytic_Simple()
s.input_selector = fade_images
## Initialize starting weights
starting_feedforward = np.exp(np.maximum(np.random.rand(s.n_neuron, s.n_stim)*0.5-0.2, 0) - 0.99)*0.05
starting_recurrent = np.zeros((s.n_neuron, s.n_neuron))
diag = np.diag(-np.matmul(starting_feedforward, starting_feedforward.T))
print(diag)
np.fill_diagonal(starting_recurrent, diag)

## Define the MNIST inputs
inputs = create_mnist_inputs(n_train_patterns, s, train=True)
test_inputs = create_mnist_inputs(n_test_patterns, s, train=False)
#fig, axs = plt.subplots()
#plot_receptor_field(axs, inputs.T, "Inputs")
#fig.savefig("Inputs.png")
#plt.close()

## Init and run net
net = SomaticNet(s)
net.feedforward_weights = starting_feedforward
net.recurrent_weights = starting_recurrent

net.run_net(inputs, test_inputs, s, update=True)
plot_mnist(folder + "/" + project + "_log")