import sys
from sklearn.model_selection import train_test_split

from plotting import *
path = '/data.nst/lucas/MB_learning/'
if path not in sys.path:
    sys.path.insert(1, path)
from classes import *
from utils import *
import numpy as np
import keras

def read_odors_csv(s):
    """
    Reads ORN mean firing rate data from csv file, stores in array. Stores baseline rates too.
    """
    assert isinstance(s, Settings)
    no_odors = s.n_odors
    no_ORN = s.n_ORN
    file_array = np.loadtxt(s.odor_file, dtype=int, skiprows=1, usecols=range(1, 1+no_ORN), delimiter=",")
    odor_matrix = file_array[:no_odors]
    baseline = file_array[-1]
    return odor_matrix, baseline

def calc_PN_rates(ORN_rates):
    """
    Calculates the PN firing rates from the ORN firing rates.
    """
    ## This data only includes 24 of the 50 ORN types
    r_tot = np.sum(ORN_rates)
    R_max = 165 ## taken from Olsen 2010, modelling section
    sigma = 12  ## taken from Olsen 2010, modelling section
    # m = 10.63 for VM7 and 4.19 for DL5 in Olsen, 2010
    m = 10.63 ## this is a mean m, but it clearly varies for diff PNs, use Olsen 2010 max out of two glomeruli
    m_eff = m/190 ## ORN total to LFP transformation, Olsen 2010
    PN = R_max*np.power(ORN_rates, 1.5)/(np.power(ORN_rates, 1.5) + sigma**1.5 + (m_eff*r_tot)**1.5)
    return PN

def calc_PN_noise(PN_rates):
    """
    Calculates the PN mean firing rate noise levels.
    """
    ## Use supplementary info of normalization paper (over entire presentation window)
    ## or Bhandawat 2007 (which seems to contradict, but is over just the spiking window)- this only is true if I'm feeding spikes as inputs
    return 9.5 - 7.2*np.exp(-np.maximum(PN_rates, 0)/76)

def generate_odor_means(s, num_data=1000, seed=0, all_odors=False, ordered=True):
    """
    Generates many noisy instantiations of PN mean firing rates for many different odors.
    """
    odors, b = read_odors_csv(s)
    noise = calc_PN_noise(odors)
    n_odors = len(odors)
    n_PN = len(odors[0])
    np.random.seed(seed)
    if all_odors:
        if ordered:
            odor_no = np.repeat(np.reshape(np.arange(n_odors), (1,-1)), num_data, axis=0)
            y = [np.random.shuffle(x) for x in odor_no]
            odor_no = odor_no.flatten()
        else:
            odor_no = np.repeat(np.arange(n_odors), num_data)
            np.random.shuffle(odor_no)
        PN_rates = np.zeros((num_data*n_odors, n_PN))
    else:
        odor_no = np.random.randint(n_odors, size=num_data)
        PN_rates = np.zeros((num_data, n_PN))

    baseline = np.zeros_like(PN_rates)
    for d in range(len(PN_rates)):
        PN_rates[d] = np.maximum(odors[odor_no[d]] + np.random.normal(0, noise[odor_no[d]]), 0)
        baseline[d] = b
    return PN_rates, baseline, odor_no

def PN_rate_dynamics(PN_mean_rates, baseline, s):
    """
    Generates dynamic PN firing rates over the odor presentation period given the mean firing rates (shape given by adaptation property of Settings file)
    """
    assert isinstance(s, Settings)
    adaptation = s.input_adaptation
    ## Assume that PN_mean_rates already has noise from Olsen SI inputted
    timesteps = s.presentation_length
    fade_fraction = s.fade_fraction
    dynamics = np.zeros((timesteps,) + PN_mean_rates.shape)
    stim_on = int(timesteps*(1-fade_fraction))
    tau_inc = s.input_tau_inc
    tau_off = s.input_tau_off
    if not adaptation:
        ## Calculate ratio of peak to mean
        PN_peak_rates = (2*stim_on*PN_mean_rates-tau_inc*baseline)/(2*stim_on-tau_inc) ## calculated by integrating
        for t in range(tau_inc):
            dynamics[t] = (PN_peak_rates-baseline)*t/tau_inc + baseline
        for t in range(tau_inc, stim_on + 1):
            dynamics[t] = PN_peak_rates
    else:
        tau_adap = s.input_tau_adap
        adap_level = s.input_adap_level
        PN_peak_rates = (timesteps*PN_mean_rates+tau_inc/2*baseline)/(((1-adap_level)*tau_adap*(1-np.exp(-(timesteps-tau_inc)/tau_adap)))+(timesteps-tau_inc)*adap_level+tau_inc/2) ## calculated by integrating
        for t in range(tau_inc):
            dynamics[t] = (PN_peak_rates-baseline)*t/tau_inc + baseline
        for t in range(tau_inc, stim_on + 1):
            dynamics[t] = PN_peak_rates*adap_level + (1-adap_level)*PN_peak_rates*np.exp(-(t-tau_inc)/tau_adap)
    final_PN_rate = dynamics[stim_on]
    for t in range(stim_on, timesteps):
        dynamics[t] = (final_PN_rate-baseline)*np.exp(-(t-stim_on)/tau_off)+baseline
    return dynamics

def PN_dynamics(PN_mean_rates, baseline, s):
    """
    Generates PN dynamics given rates, and adds spiking noise if necessary.
    """
    ## Noise is of two forms: mean noise that varies among odor presentations, and spike noise within each presentation
    assert isinstance(s, Settings)
    rate_dyn = PN_rate_dynamics(PN_mean_rates, baseline, s)
    if s.input_spiking_noise:
        sample_duration = 50 ## ms TODO: ask Lucas about changing this
        sample_interval = int(sample_duration/s.dt) ##
        spike_dyn = np.zeros_like(rate_dyn)
        timesteps = len(rate_dyn)
        for t in range(0, timesteps, sample_interval):
            spike_dyn[t] = np.random.normal(loc=rate_dyn[t]*sample_duration/1000, scale=0.5)
            for t2 in range(t, t+sample_interval):
                spike_dyn[t2] = spike_dyn[t]
            if s.input_spiking_memory and t > 0:
                for t2 in range(t-sample_interval, t):
                    spike_dyn[t2] = (1-(t2-t+sample_interval)/sample_interval)*spike_dyn[t-sample_interval] + (t2-t+sample_interval)/sample_interval*spike_dyn[t]
        return spike_dyn                
    else:
        return rate_dyn

def get_PN_dynamics(s, num_data=1000, seed=0, all_odors=False, return_odor_no=False, scaled=True):
    """
    Generates PN dynamics for a given number of odor presentations.
    """
    assert isinstance(s, Settings)
    s.set_dt() # make sure that time units are in the right units after initialization
    PN_mean, b, odors = generate_odor_means(s, num_data=num_data, seed=seed, all_odors=all_odors)
    if s.input_baseline:
        base = b
    else:
        base = np.zeros_like(PN_mean)
    dynamics = PN_dynamics(PN_mean, base, s)
    if scaled:
        dynamics = dynamics/200 ## 200 assumed to be peak firing rate
    #PN_no = dynamics.shape[-1]
    dynamics = np.swapaxes(dynamics, 0, 1)
    if return_odor_no:
        return dynamics, odors
    else:
        return dynamics


def generate_KC_rates(net, s, odor_file, odor_repeats=10, seed=0, odor_averaged=False):
    assert isinstance(s, Settings)
    assert isinstance(net, Net)
    classifier_train_odors, odor_no = get_PN_dynamics(s, num_data=odor_repeats, return_odor_no=True, all_odors=True, seed=seed)
    z, r = net.eval_net(classifier_train_odors, s)
    if odor_averaged:
        z = np.mean(z, axis=1)
        r = np.mean(r, axis=1)
        odor_label = odor_no
    else:
        odor_label = np.repeat(odor_no, int(s.presentation_length*(1-s.fade_fraction))) ## Make a list of labels for every KC rate output seen

    odor_z = z.reshape((-1, net.n_neuron)) ## neuron potentials for each timestep that odor is on
    odor_r = r.reshape((-1, net.n_neuron))
    return odor_z, odor_r, odor_label

def train_odor_classifier(KC_rates, odor_labels, num_odors=110, batchsize=100, epochs=10):
    ## Returns a 1-layer neural net with nonlinear activation function (e.g. sigmoid) on linear combination of inputs
    ## Needs to have non-negative weights
    num_KC = len(KC_rates[0])
    one_hot_odor = keras.utils.to_categorical(odor_labels, num_odors)
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units=num_odors, input_shape=(num_KC,), activation='softmax', kernel_constraint=keras.constraints.nonneg())) ## without activation
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(KC_rates, one_hot_odor, batch_size=batchsize, epochs=epochs, verbose=1, validation_split=.3)
    return model, history.history

def confusion_matrix(net, s, odor_file, model, odor_repeats=10, return_r=False, odor_averaged=False):
    ## Take the model from train, and then feed it inputs from n_odors odors through eval_net, take average of probabilities over timesteps and odor trials
    odor_z, odor_r, odor_label = generate_KC_rates(net, s, odor_file, odor_repeats=odor_repeats, seed=1, odor_averaged=odor_averaged)
    predictions = model.predict(odor_z)
    n = s.n_odors
    confusion = np.zeros((n,n))
    for i in range(n):
        confusion[:, i] = np.mean(predictions[np.where(odor_label == i)] , axis=0)
    if return_r:
        return confusion, odor_r
    return confusion

def train_net_classifier(net, s, odor_file, odor_repeats=10, epochs=100, odor_averaged=False):
    """
    Train a classifier to classify the odor based on the outputs of the net inputted.
    """
    odor_z, odor_r, odor_label = generate_KC_rates(net, s, odor_file, odor_repeats=odor_repeats, odor_averaged=odor_averaged)
    model, hist = train_odor_classifier(odor_z, odor_label, num_odors=s.n_odors, epochs=epochs)
    accuracy = hist['val_accuracy'][-1]
    return model, hist, accuracy

def analyze_net(net, s, odor_file, odor_repeats, save_info=True, plot_info=True, epochs=100, odor_averaged=False):
    model, hist_dict, acc = train_net_classifier(net, s, odor_file=odor_file, odor_repeats=odor_repeats, epochs=epochs,  odor_averaged=odor_averaged)
    print("Accuracy for learned network is", acc)
    confusion, r = confusion_matrix(net, s, odor_file, model, odor_repeats=odor_repeats, return_r=True, odor_averaged=odor_averaged)
    net_analysis = {"Model": model, "Hist": hist_dict, "Confusion": confusion, "Rates": r, "Accuracy": acc}
    if save_info:
        with gzip.open('%s/%s_info' % (s.folder_name, s.project_name), 'wb') as f:
            pickle.dump(net_analysis, f, protocol=2)
    if plot_info:
        plot_odor_full('%s/%s_log' % (s.folder_name, s.project_name), r, hist_dict, confusion)
    return net_analysis

class MBON():

    def __init__(self, s):
        self.n_KC = s.n_neuron
        self.settings = s.MBON_settings_dict
        self.n_MBON = 1
        self.weights = np.zeros((self.n_MBON, self.n_KC))
    
    def compute_DAN_input(self, valence):
        pass

    def compute_valence(self, KC_z):
        pass

    def learn_weights(self, KC_z, valence):
        pass

    def train(self, KC_zs, valences, epochs=1):
        ntrain = len(KC_zs)
        for e in range(epochs):
            for t in range(ntrain):
                self.learn_weights(KC_zs[t], valences[t])
    
    def test(self, KC_zs, valences):
        ntest = len(KC_zs)
        accuracy = np.zeros((ntest,))
        pred = np.zeros_like(accuracy)
        for t in range(ntest):
            pred[t] = self.compute_valence(KC_zs[t])
            accuracy[t] = abs(pred[t] - valences[t])
        return np.mean(accuracy)

    def train_converge(self, KC_zs, valences, error=0.05):
        prior_acc = 0
        while True:
            self.train(KC_zs, valences)
            new_acc = self.test(KC_zs, valences)
            if (abs(new_acc - prior_acc))/prior_acc < error:
                break

class Predictive_MBON(MBON):
    
    def __init__(self, s):
        super().__init__(s)
        self.n_MBON = 1
        self.weights = np.zeros((self.n_MBON, self.n_KC))

    def compute_valence(self, KC_z):
        return np.dot(self.weights[0], KC_z)
    
    def compute_DAN_input(self, valence, KC_z):
        return valence - self.compute_valence(KC_z)
    
    def learn_weights(self, KC_z, valence):
        eta_learning = self.settings["learning_rate"]
        self.weights[0] += eta_learning * (self.compute_DAN_input(valence, KC_z) - self.compute_valence(KC_z)) * KC_z
        ## the problem is this assumes same number of non-zero KC_zs for each odor (since paper has 1 odor, 1 KC)
        ## bc otherwise no way to normalize weights and therefore valences. But could just use sign ig?
    
class Approach_Avoid(MBON):
    def __init__(self, s):
        super().__init__(s)
        self.n_MBON = 2
        ## Approach is weights[0], avoid is weights[1]
        self.weights = np.ones((self.n_MBON, self.n_KC))
    
    def compute_DAN_input(self, valence):
        return valence

    def compute_valence(self, KC_z):
        c = self.settings["c"]
        return np.exp(c*np.dot(self.weights[1], KC_z)) / (np.exp(c*np.dot(self.weights[0], KC_z)) + np.exp(c*np.dot(self.weights[1], KC_z)))
    
    def learn_weights(self, KC_z, valence):
        eta_learning = self.settings["learning_rate"]
        target_MBON = 1 - self.compute_DAN_input(valence)
        self.weights[target_MBON] += np.multiply((np.exp(-eta_learning * KC_z) - 1), self.weights[target_MBON])
    
    def test(self, KC_zs, valences):
        ntest = len(KC_zs)
        accuracy = np.zeros((ntest,))
        pred = np.zeros_like(accuracy)
        for t in range(ntest):
            pred[t] = self.compute_valence(KC_zs[t])
            ## can speed up by doing this all in one
        behav = np.where(pred > np.random.rand(ntest), 1, 0)
        accuracy = 1 - np.count_nonzero(behav-valences)/ntest
        return accuracy

def analyze_net_valence(net, s, odor_file, odor_repeats, odor_averaged=False):
    odor_z, odor_r, odor_label = generate_KC_rates(net, s, odor_file, odor_repeats=odor_repeats, odor_averaged=odor_averaged)
    valences = np.where(odor_label > s.n_odors/2, 1, 0) ## classify some odors into approach/avoid
    KC_train, KC_test, valence_train, valence_test = train_test_split(odor_z, valences, test_size=0.3)
    mbon = Approach_Avoid(s)
    mbon.train(KC_train, valence_train)
    accuracy = mbon.test(KC_test, valence_test)
    print("MBON Accuracy:", accuracy)
    return accuracy

def learning_speed(net, s, odor_file, odor_repeats, odor_averaged=False, interval=None, plateau=15, plot=False):
    assert isinstance(s, Settings)
    if interval is None:
        interval = 1
    if plateau > odor_repeats:
        plateau = odor_repeats
    training_repeats = [x for x in range(1, plateau, interval)] + [int(odor_repeats/2), odor_repeats]
    mbon_accuracy = []
    clas_accuracy = []
    odor_z, odor_r, odor_label = generate_KC_rates(net, s, odor_file, odor_repeats=odor_repeats, odor_averaged=odor_averaged)
    valences = np.where(odor_label > s.n_odors/2, 1, 0) ## classify some odors into approach/avoid
    one_hot_odor = keras.utils.to_categorical(odor_label, s.n_odors)
    for idx, repeat in enumerate(training_repeats):
        trials = int(len(odor_z)*repeat/odor_repeats)
        subset_z, subset_label, subset_valence = odor_z[:trials], one_hot_odor[:trials], valences[:trials]
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(units=s.n_odors, input_shape=(s.n_neuron,), activation='softmax', kernel_constraint=keras.constraints.nonneg())) ## without activation
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(subset_z, subset_label, epochs=20, verbose=0, validation_split=.3)
        mbon = Approach_Avoid(s)
        KC_train, KC_test, valence_train, valence_test = train_test_split(subset_z, subset_valence, test_size=0.3)
        mbon.train(KC_train, valence_train, epochs=max(1, int(10/repeat)))
        mbon_acc = mbon.test(KC_test, valence_test)
        clas_accuracy.append(history.history['val_accuracy'][-1])
        mbon_accuracy.append(mbon_acc)
        print("Unique Odors:", repeat)
        print("NN Classifier Accuracy: ", clas_accuracy[-1])
        print("MBON Valence Accuracy: ", mbon_accuracy[-1])
    accdict = {"Repeats": training_repeats, "MBON": mbon_accuracy, "Classifier": clas_accuracy}
    if plot:
        confusion, r = confusion_matrix(net, s, odor_file, model, odor_repeats=odor_repeats, return_r=True, odor_averaged=odor_averaged)
        net_analysis = {#"Model": model, "Hist": history,
        "Confusion": confusion, "Rates": r, "Accuracy": accdict}
        plot_odor_full('%s/%s_log' % (s.folder_name, s.project_name), r, history.history, confusion, accdict, PN_rate_dynamics(np.array([[1]]), 0.05, s))
        with gzip.open('%s/%s_info' % (s.folder_name, s.project_name), 'wb') as f:
            pickle.dump(net_analysis, f, protocol=2)
    return accdict

def read_meas_connectivity(meas_file):
    file_array = np.loadtxt(meas_file, dtype=int, skiprows=1, delimiter=",")
    return file_array

def map_sim_to_meas_connectivity(sim_net, meas, s):
    """
    Returns a mapping of neurons from the simulated net to the measured net, based on their input similarities."
    """
    assert isinstance(s, Settings)
    assert isinstance(sim_net, Net)
    best_mapping = np.zeros((s.n_neuron,))
    mapping_MSE = np.zeros_like(best_mapping)
    for KC_idx in range(s.n_neuron):
        sim_weights_mat = np.reshape(np.tile(sim_net.feedforward_weights[KC_idx], s.n_neuron), (-1, s.n_neuron))
        MSE = np.sum(np.square(sim_weights_mat-meas), axis=1)
        best_neuron = np.argmin(MSE)
        best_mapping[KC_idx] = best_neuron
        mapping_MSE[KC_idx] = np.amin(MSE)
    return best_mapping, mapping_MSE

def compute_KC_feedforward_correlations(connectivity, s):
    """
    Compute correlations between columns of the connectivity matrix - i.e. are the inputs from PN1, PN2 correlated across KCs?
    """
    assert isinstance(s, Settings)
    corr = np.zeros((s.n_stim, s.n_stim))
    for PN1 in range(s.n_stim):
        for PN2 in range(s.n_stim):
            corr[PN1, PN2] = np.corrcoef(connectivity[:, PN1], connectivity[:, PN2])[0, 1]
    return corr


def compute_PN_firing_correlations(s):
    """
    Compute correlations between firing rates of different PNs across odors
    """
    assert isinstance(s, Settings)
    odor_data, b = read_odors_csv(s)
    corr = np.zeros((s.n_stim, s.n_stim))
    for PN1 in range(s.n_stim):
        for PN2 in range(s.n_stim):
            corr[PN1, PN2] = np.corrcoef(odor_data[:, PN1], odor_data[:, PN2])[0, 1]
    return corr

def compute_PN_input_correlations(connectivity, s):
    """
    Weighted (by product of connection strength) average of the correlations between PNs that connect to each KC (would expect high in DB)
    
    """
    ## Connectivity has dimensions (KC, PN)
    assert isinstance(s, Settings)
    PN_corr = compute_PN_firing_correlations(s)
    KC_input_corr = np.zeros((s.n_neuron,))
    assert isinstance(s, Settings)
    for KC_idx in range(s.n_neuron):
        weights = np.reshape(connectivity[KC_idx], (-1, 1))
        weight_products = np.fill_diagonal(np.matmul(weights, weights.T), 0)
        weighted_corr = np.multiply(weight_products, PN_corr)
        weighted_avg = np.sum(weighted_corr)/np.sum(weight_products)
        KC_input_corr[KC_idx] = weighted_avg
    return KC_input_corr

def compute_input_output_correlations(connectivity, s):
    """
    Correlation of the correlation between the firing rates of two PNs and the correlation between their input strength to each KC.   
    """
    assert isinstance(s, Settings)
    idxs = np.triu_indices(s.n_stim)
    PN_corr = compute_PN_firing_correlations(s)[idxs]
    KC_corr = compute_KC_feedforward_correlations(connectivity, s)[idxs]
    corr = np.corrcoef(PN_corr, KC_corr)[0,1]
    return corr

