import sys
path = '/data.nst/arana/olfaction_circuitry/'
if path not in sys.path:
    sys.path.insert(1, path)
from classes import *
from odor_input import *
from utils import *

## Define the files containing the data to be extracted
folder_name = path + "odor_test/simulations"
project_name = "odor_0713_distinct_long"
odor_file = '/data.nst/arana/olfaction_circuitry/odor_test/odor_distinct.csv'

log_file = folder_name+"/"+project_name + "_log"
with gzip.open(log_file, 'r') as f:
    log = pickle.load(f)
assert isinstance(log, Log)
final_time = max(log.snapshots.keys())
final_snap = log.snapshots[final_time]
s = final_snap["s"]
print("Odors Trained on:", int(final_time/s.presentation_length))

## Define the odor input files
n_test_odor_repeats = 10
net = SomaticNet(s)
net.feedforward_weights = final_snap["feedforward_weights"]
net.recurrent_weights = final_snap["recurrent_weights"]
net.neuron_biases = final_snap["biases"]
net.decoder_matrix = final_snap["decoder"]
analyze_net(net, s, odor_file, n_test_odor_repeats, save_info=False, plot_info=True, epochs=10)