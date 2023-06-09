import time
import pickle
import gzip
import numpy as np

class Neuron():
    def __init__(self, s, seed=0):
        self.kernel_length = s.kernel_length
        self.kernel = s.kernel
        self.t = 0 ## timesteps (not actual time)
        self.recent_spikes = []
        self.rng = np.random.default_rng(seed=seed)

    def check_spike(self, input):
        ## Keep recent_spikes updated
        if len(self.recent_spikes) > 0 and (self.t - self.recent_spikes[0] >= self.kernel_length):
            self.recent_spikes.pop(0)
        ## Spike stochastically
        p = 1/(1 + np.exp(-input))
        r = self.rng.random()
        if r < p:
            self.recent_spikes.append(self.t)
        self.t += 1
        return r < p

    def get_psp(self):
        ## Including a 1-timestep delay; only spikes from >= 1 timestep ago are counted here
        out = 0
        for spike in self.recent_spikes:
            time_after_spike = self.t - spike - 1 ## 1 here is the timestep delay
            if time_after_spike < 0:
                continue
            out += self.kernel[time_after_spike]
        return out

class Net():
    def __init__(self, s, seed=0):
        assert isinstance(s, Settings)
        s.set_dt() #important! Transforms all variables with time dimension in units of dt
        self.n_stim = s.n_stim
        self.stim_x = np.zeros((self.n_stim, 1))
        self.kernel_length = s.kernel_length
        self.kernel = s.kernel
        self.timestep_delay = s.timestep_delay

        self.n_neuron = s.n_neuron
        #self.neurons = [Neuron(s, seed=i) for i in range(self.n_neuron)]
        self.neuron_psps = np.zeros((self.n_neuron,))
        self.membrane_potentials = np.zeros_like(self.neuron_psps)
        self.membrane_thresholds = np.zeros_like(self.neuron_psps)

        self.neuron_spikes = np.zeros_like(self.neuron_psps, dtype=bool)
        self.recent_spikes = np.zeros((self.kernel_length + self.timestep_delay, self.n_neuron), dtype=bool)
        self.neuron_rates = np.zeros_like(self.neuron_psps) ## average spike rate per neuron per timestep, reset at each batch
        self.neuron_biases = np.ones_like(self.neuron_psps)*np.log(1.0 / (1 - s.rho) - 1.0)
        self.decoder_matrix = np.zeros((self.n_stim, self.n_neuron))
        self.feedforward_weights = np.zeros((self.n_neuron, self.n_stim))
        self.recurrent_weights = np.zeros((self.n_neuron, self.n_stim, self.n_neuron))

        self.rng = np.random.default_rng(seed=seed)

        self.sigma = s.initial_sigma
        self.plasticity = s.plasticity

        self.reconstruction = np.zeros((self.n_stim, 1))
        self.memory = {"last_spike_time": 0, "last_spike_indices": [], "last_z": np.zeros((self.n_neuron,)), "last_x": np.zeros((self.n_stim,))}
        self.log = Log()
        self.s = s
    
    def output_net(self):
        return {'F': self.feedforward_weights, 'D': self.decoder_matrix, 'W': self.recurrent_weights, 'b': self.neuron_biases, 's': self.s}

    def eval_net(self, inputs, s):
        assert isinstance(s, Settings)
        l = s.presentation_length
        f = s.fade_fraction
        select_input = s.input_selector
        n_inp = len(inputs)
        n_steps = n_inp * l
        store_steps = int(l*(1-f))
        z = np.zeros((len(inputs), store_steps, self.n_neuron))
        r = np.zeros_like(z, dtype=bool)
        now = time.time()
        for tim in range(n_steps):
            if tim % 50000 == 0:
                inp = int(tim/l)
                progress = inp/n_inp
                print("Input: %s of %s" % (inp, n_inp), f"{progress:.1%}", flush=True)
            x = select_input(inputs, tim, s)
            self.step_net(x, s, update=False)
            t = tim % l
            if t < l*(1-f):
                z[int(tim/l), t, :] = self.neuron_psps
                r[int(tim/l), t, :] = self.neuron_spikes
        return z, r

    def run_net(self, inputs, test_inputs, s, update=True, decoder_only=False, homeostatic_bias=False):
        assert isinstance(s, Settings)
        select_input = s.input_selector
        l = s.presentation_length
        n_inp = len(inputs)
        n_steps = n_inp * l
        self.log.setup_dynamic_log(s, n_steps)
        for time in range(n_steps):
            if time % 100000 == 0:
                inp = int(time/l)
                progress = inp/n_inp
                print("Input: %s of %s" % (inp, n_inp), f"{progress:.1%}", flush=True)
            self.log.t = time
            x = select_input(inputs, time, s)
            batch = time % s.update_interval == 0
            self.step_net(x, s, update_batch=batch, update=update, decoder_only=decoder_only, homeostatic_bias=homeostatic_bias)
            self.log.log_results(self, test_inputs, s)
        self.log.take_snapshot(self, test_inputs, s)
        self.log.write_status(s)

    # def step_net_old(self, x_input, s, update_batch=False, update=True, decoder_only=False):
    #     self.stim_x = x_input
    #     inputs = self.calc_inputs()
    #     for neuron in range(self.n_neuron):
    #         self.neuron_spikes[neuron] = self.neurons[neuron].check_spike(inputs[neuron])
    #     if update:
    #         self.plasticity.update_net(self, s, update_batch, decoder_only=decoder_only)
    #     for neuron in range(self.n_neuron):
    #         self.neuron_psps[neuron] = self.neurons[neuron].get_psp()
    
    def step_net(self, x_input, s, update_batch=False, update=True, decoder_only=False, homeostatic_bias=False):
        self.stim_x = x_input
        inputs = self.calc_inputs()
        probability = 1/(1 + np.exp(-inputs))
        self.neuron_spikes = np.where(self.rng.random(self.n_neuron) < probability, True, False)
        self.neuron_rates += self.neuron_spikes / s.update_interval
        self.recent_spikes[1:] = self.recent_spikes[:-1]
        self.recent_spikes[0] = self.neuron_spikes
        if update:
            self.plasticity.update_net(self, s, update_batch, decoder_only=decoder_only, homeostatic_bias=homeostatic_bias)
        self.neuron_psps = np.matmul(self.recent_spikes[self.timestep_delay:].T, self.kernel)

    def calc_inputs(self):
        pass

class DendriticNet(Net):
    def __init__(self, s):
        super().__init__(s)
        self.recurrent_weights = np.zeros((self.n_neuron, self.n_stim, self.n_neuron))
        self.dendritic_potentials = np.zeros((self.n_stim, self.n_neuron))

    def calc_inputs(self):
        for neuron in range(self.n_neuron):
            self.dendritic_potentials[:, neuron] = np.multiply(self.feedforward_weights[neuron], self.stim_x) \
                                                   + np.dot(self.recurrent_weights[neuron, :, :], self.neuron_psps)
            self.membrane_potentials = np.sum(self.dendritic_potentials, axis=0)
            self.membrane_thresholds[neuron] = - 0.25*self.recurrent_weights[neuron, :, neuron] \
                                               - np.square(self.sigma)*self.neuron_biases[neuron]
        spike_inputs = (self.membrane_potentials - self.membrane_thresholds)*np.reciprocal(np.square(self.sigma))
        return spike_inputs

class SomaticNet(Net):
    def __init__(self, s):
        super().__init__(s)
        self.recurrent_weights = np.zeros((self.n_neuron, self.n_neuron))

    def calc_inputs(self):
        self.membrane_potentials = np.matmul(self.feedforward_weights, self.stim_x) \
                                                   + np.matmul(self.recurrent_weights, self.neuron_psps)
        self.membrane_thresholds = - 0.25*np.diagonal(self.recurrent_weights) \
                                               - np.square(self.sigma)*self.neuron_biases
        spike_inputs = (self.membrane_potentials - self.membrane_thresholds)*np.reciprocal(np.square(self.sigma))
        return spike_inputs

class Decoder():
    def reconstruct_input(self, net):
        return np.matmul(net.decoder_matrix, net.neuron_psps)

    def decoder_loss(self, net):
        err = net.stim_x - self.reconstruct_input(net)
        loss = (0.5/net.n_stim) * np.dot(err, err)
        return loss

    def update_decoder(self, net, s):
        pass

class OptimalDecoder(Decoder):
    def update_decoder(self, net, s):
        z_col = net.neuron_psps.reshape((net.n_neuron, 1))
        dD = np.matmul(net.stim_x.reshape(-1, 1) - np.matmul(net.decoder_matrix, z_col), z_col.T)
        net.decoder_matrix += s.eta_decoder * dD

    def sparsify_decoder(self, net, s):
        sum_over_i = np.sum(D, axis=0)



class OptimalDecoderEfficient(Decoder):
    def update_decoder(self, net, s):
        tau = s.kernel_tau

        t = net.log.t
        tl = net.memory["last_spike_time"]

        sum1 = np.sum([np.exp((tl - ts + 1) / tau) for ts in range(tl + 1, t + 1)])  # tau * (1 - exp((tl - t) / tau))
        sum2 = np.sum(
            [np.exp(2 * (tl - ts + 1) / tau) for ts in range(tl + 1, t + 1)])  # 0.5 * tau * (1 - exp(2*(tl - t) / tau))

        z_col = net.memory["last_z"].reshape((net.n_neuron, 1))
        x_row = net.memory["last_x"].reshape((1, net.n_stim))

        dD = np.matmul((x_row.T * sum1) - (np.matmul(net.decoder_matrix, z_col) * sum2), z_col.T)
        net.decoder_matrix += s.eta_decoder * dD


class Plasticity():

    decoder = OptimalDecoder()

    #def __init__(self):
    #    self.decoder = OptimalDecoder()

    def change_decoder(self, decoder):
        self.__class__.decoder = decoder

    def update_net(self, net, s, update_batch=False, decoder_only=False):
        pass

    def update_feedforward(self, net, s):
        pass

    def update_recurrent(self, net, s):
        pass

    def update_bias(self, net, s):
        eta = s.update_interval * s.eta_bias
        goalrate = s.rho ## firing rate per timestep
        net.neuron_biases += eta * (goalrate - net.neuron_rates)
        #print(goalrate, net.neuron_rates)
        net.neuron_rates = np.zeros_like(net.neuron_psps)

class Analytic_Simple(Plasticity):

    def update_net(self, net, s, update_batch=False, decoder_only=False, homeostatic_bias=False):
        if decoder_only:
            self.calculate_recurrent(net)
        else:
            self.update_feedforward(net, s)
            self.update_recurrent(net, s)
        self.decoder.update_decoder(net, s)
        if homeostatic_bias:
            if update_batch:
                self.update_bias(net, s)

    def update_feedforward(self, net, s):
        z_col = net.neuron_psps.reshape((net.n_neuron,1))
        x_row = net.stim_x.reshape((1, net.n_stim))
        dF = np.matmul(z_col, x_row - np.matmul(z_col.T, net.feedforward_weights))
        net.feedforward_weights += s.eta_feedforward * dF

    def update_recurrent(self, net, s):
        net.recurrent_weights -= s.eta_recurrent * np.matmul(net.membrane_potentials.reshape((net.n_neuron,1)),
                                                 net.neuron_psps.reshape((1, net.n_neuron)))
    
    def calculate_recurrent(self, net):
        net.recurrent_weights = - np.matmul(net.feedforward_weights, net.decoder_matrix)

class Analytic_Efficient(Plasticity):

    def update_net(self, net, s, update_batch=False, decoder_only=False):
        assert isinstance(s, Settings)
        if decoder_only:
            self.decoder.update_decoder(net, s)
            return None
        if update_batch:
            if s.learned_recurrent:
                self.update_recurrent(net, s)
            self.update_bias(net, s)

        if np.sum(net.neuron_spikes) > 0:
            self.decoder.update_decoder(net, s)
            self.update_feedforward(net, s)
            if not s.learned_recurrent:
                self.calculate_recurrent(net)

            net.memory["last_spike_time"] = net.log.t
            net.memory["last_x"] = net.stim_x
            net.memory["last_z"] = net.neuron_psps*np.exp(-1/s.kernel_tau) + net.neuron_spikes
            #net.neuron_rates += net.neuron_spikes / s.update_interval

    def update_feedforward(self, net, s):
        ## Integrate learning rule over the time between last spike and current spike, making use of exponential kernel
        tau = s.kernel_tau

        t = net.log.t
        tl = net.memory["last_spike_time"]

        sum1 = np.sum([np.exp((tl - ts + 1) / tau) for ts in range(tl + 1, t + 1)])  # tau * (1 - exp((tl - t) / tau))
        sum2 = np.sum([np.exp(2 * (tl - ts + 1) / tau) for ts in range(tl + 1, t + 1)])  # 0.5 * tau * (1 - exp(2*(tl - t) / tau))

        z_col = net.memory["last_z"].reshape((net.n_neuron, 1))
        x_row = net.memory["last_x"].reshape((1, net.n_stim))

        dF = np.matmul(z_col, (x_row * sum1) - (np.matmul(z_col.T, net.feedforward_weights) * sum2))
        net.feedforward_weights += s.eta_feedforward * dF

    def update_recurrent(self, net, s):
        net.recurrent_weights -= s.eta_recurrent * np.matmul(net.membrane_potentials.reshape((net.n_neuron, 1)),
                                                 net.neuron_psps.reshape((1, net.n_neuron)))

    def calculate_recurrent(self, net):
        net.recurrent_weights = - net.feedforward_weights * net.decoder_matrix

class Log():
    def __init__(self):
        self.snapshots = dict()
        self.dynamic_log = {}
        self.running_log = {}
        self.t = 0

    def setup_dynamic_log(self, s, n_steps):
        for var in s.dynamic_vars:
            self.dynamic_log[var] = np.zeros((int(n_steps/s.dynamic_log_interval)),)
            self.running_log[var] = 0
        self.dynamic_log["t"] = np.arange(0, n_steps, s.dynamic_log_interval)

    def update_running_log(self, net, s):
        num_samples = s.dynamic_log_interval/s.dynamic_sample_interval
        for var in s.dynamic_vars:
            if var[:4] == "mean":
                exec("self.running_log[var] += np.mean(net.%s) / num_samples" % var[5:])
            elif var[:3] == "std":
                exec("self.running_log[var] += np.std(net.%s) / num_samples" % var[5:])

    def log_dynamic_log(self, net, test_inputs, s):
        assert isinstance(s, Settings)
        assert isinstance(net, Net)
        select_input = s.input_selector
        t = net.log.t
        k = int(t/s.dynamic_log_interval)
        for var in s.dynamic_vars:
            self.dynamic_log[var][k] = self.running_log[var]
            self.running_log[var] = 0
        if test_inputs is not None:
            l = s.presentation_length
            log_test_len = s.dynamic_log_test_set
            n_test_steps = log_test_len*l
            decoder_loss = 0
            for test_t in range(n_test_steps):
                test_x = select_input(test_inputs, test_t, s)
                net.step_net(test_x, s, update=False)
                if (test_t % l) < (l * (1-s.fade_fraction)):
                    decoder_loss += net.plasticity.decoder.decoder_loss(net)
            self.dynamic_log["test_decoder_loss"][k] = decoder_loss/(n_test_steps*(1-s.fade_fraction))

    def log_results(self, net, test_inputs, s):
        interval = s.dynamic_log_interval
        self.running_log["avg_firing_rate"] += np.average(net.neuron_spikes) / (interval*s.dt) * 1000 ## contains mean firing rate (spikes per neuron in Hz) across all neurons over interval
        if self.t % s.dynamic_sample_interval == 0:
            self.update_running_log(net, s)
        if self.t % interval == 0:
            self.log_dynamic_log(net, test_inputs, s)
        if self.t in s.snapshot_times:
            self.take_snapshot(net, test_inputs, s)
            self.write_status(s)

    def take_snapshot(self, net, test_inputs, s):
        select_input = s.input_selector
        l = s.presentation_length
        fade_length = s.fade_fraction
        interval = s.snapshot_log_interval
        images = len(test_inputs)
        n_steps = len(test_inputs)*l
        n_intervals = int(n_steps/interval)
        snapshot = dict()
        snapshot["recurrent_weights"] = net.recurrent_weights
        snapshot["feedforward_weights"] = net.feedforward_weights
        snapshot["decoder"] = net.decoder_matrix
        snapshot["biases"] = net.neuron_biases
        snapshot["reconstructions"] = np.zeros((images, l, net.n_stim))
        snapshot["reconstruction_means"] = np.zeros((images, net.n_stim))
        snapshot["reconstruction_vars"] = np.zeros((images, net.n_stim))
        snapshot["input_means"] = np.zeros_like(snapshot["reconstruction_means"])
        snapshot["spikes"] = np.zeros((n_intervals, net.n_neuron), dtype=bool)
        snapshot["inputs"] = test_inputs
        snapshot["s"] = s
        neuron_spikes = np.zeros((net.n_neuron), dtype=bool)
        for t in range(n_steps):
            x = select_input(test_inputs, t, s)
            net.step_net(x, s, update=False)
            rec = net.plasticity.decoder.reconstruct_input(net)
            snapshot["reconstructions"][int(t/l), t % l, :] = rec
            if (t % l) < (l * (1-fade_length)):
                snapshot["input_means"][int(t/l), :] += x / (l * (1.0 - fade_length))
                snapshot["reconstruction_means"][int(t/l), :] += rec / (l * (1.0 - fade_length))
                snapshot["reconstruction_vars"][int(t / l), :] += np.square(net.stim_x - rec) / (l * (1.0 - fade_length))
            neuron_spikes = np.logical_or(neuron_spikes, net.neuron_spikes)
            if t % interval == 0:
                ind = int(t / interval)
                snapshot["spikes"][ind, :] = neuron_spikes
                neuron_spikes = np.zeros((net.n_neuron), dtype=bool)
        self.snapshots[self.t] = snapshot

    def save_log(self, s):
        assert isinstance(s, Settings)
        folder = s.folder_name
        project = s.project_name
        with gzip.open('%s/%s_log' % (folder, project), 'wb') as f:
            pickle.dump(self, f, protocol=2)

    def write_status(self, s):
        self.save_log(s)


class Settings():
    def __init__(self):
        ## Timesteps
        self.dt = 1 ## ms
        self.timestep_delay = 1 # in steps!
        self.time_unit = "ms"
    
        ## Input dynamics
        self.presentation_length = 100  ## ms 
        self.fade_fraction = 0.2  ## denotes fraction of presentation fading to next step

        ## Odor specific inputs
        self.n_odors = 110
        self.n_ORN = 24

        ## Odor specific input dynamics
        self.input_adaptation = True
        self.input_baseline = True
        self.input_tau_inc = 100 ## time (in ms) to reach peak firing 
        self.input_tau_off = 200 ## timescale (in ms) of decay after odor is switched off (during fade)
        self.input_tau_adap = 150 ## timescale (in ms) of adaptation to adapted level after peak is hit
        self.input_adap_level = 0.5 ## fraction of peak firing level that input adapts to
        self.input_spiking_noise = False
        self.input_spiking_memory = True
        self.stimulus_strength_on = 1.0
        self.stimulus_strength_off = 0.0
        self.input_selector = None
        self.n_test_repeat = 0
        self.n_train_repeat = 0

        ## Network architecture
        self.n_stim = np.nan
        self.n_neuron = np.nan

        ## Neuronal dynamics
        ## set using set_dt to be implemented still
        self.kernel_tau = 10  # time constant of PSP decay, 10ms
        self.kernel_length = 5 * self.kernel_tau
        self.kernel = np.exp(-np.arange(self.kernel_length) / self.kernel_tau)
        # target firing rate across neurons (spikes/ms)
        self.rho = 0.02

        ## For generating the random initial params
        self.recurrent_variance = 0.0
        self.recurrent_mean = 0.0

        self.initial_sigma = np.sqrt(0.1) ## starting sigma used for learning; noise of the decoder

        ## Plasticity rules
        self.plasticity = Plasticity
        self.learned_recurrent = True  ## Recurrence is learned or calculated from feedforward?
        self.update_interval = 5  ## update interval (ms) that batch updates are processed in

        ## NN plasticity learning settings
        self.eta_feedforward = 0.00006 # 1/ms
        self.eta_recurrent = 0.00006 #1/ms
        self.eta_decoder = 0.0001 # 1/ms
        self.eta_bias = 0.00001 # 1/ms

        ## Sigma annealing
        self.learned_sigma = False
        self.eta_sigma = 0 # 1/ms
        self.fixed_final_sigma = self.initial_sigma

        ## Logging settings
        self.dynamic_log_interval = 5000  ## stores dynamic variable averages every ___ ms; length of averaging window
        self.dynamic_sample_interval = 2.5  # samples the dynamic variables of the net every ___ ms
        self.snapshot_log_interval = self.dt  # saves only every ___ ms time point in the snapshot to save space
        self.dynamic_vars = [ #"mean_feedforward_weights", "mean_recurrent_weights", "neuron_biases",
                             "avg_firing_rate",  "test_decoder_loss"]
        self.dynamic_log_test_set = self.n_odors ## fraction of test inputs for snapshot used in dynamic log
        self.snapshot_times = []
        self.folder_name = ""
        self.project_name = ""
        self.odor_file = ""
        self.description = ""

        ## MBON learning
        self.MBON_settings_dict = {"c": 1, "learning_rate": 0.00001}


    def set(self, key, value):
        # if key == 'dt':
        #     self.set_dt(value)
        # else:
        exec("self.%s = %s" % (key, value))

    def set_dt(self):
        if self.time_unit != "steps":
            dt = self.dt
            self.presentation_length = int(self.presentation_length/dt)  ## num steps
            self.input_tau_inc = int(self.input_tau_inc/dt)  ## num steps
            self.input_tau_adap = int(self.input_tau_adap/dt)  ## num steps
            self.input_tau_off = int(self.input_tau_off/dt)  ## num steps
            self.kernel_tau = int(self.kernel_tau/dt)  # [step]
            self.kernel_length = int(self.kernel_length/dt)
            self.kernel = np.exp(-np.arange(self.kernel_length) / self.kernel_tau)
            self.eta_decoder =  self.eta_decoder * dt # [1/step]
            self.eta_feedforward =  self.eta_feedforward * dt # [1/step]
            self.eta_recurrent =  self.eta_recurrent * dt # [1/step]
            self.eta_bias =  self.eta_bias * dt # [1/step]
            self.eta_sigma = self.eta_sigma * dt # [1/step]
            self.rho = self.rho * dt # transform to [1/steps]
            self.dynamic_log_interval = int(self.dynamic_log_interval/dt)
            self.dynamic_sample_interval = int(self.dynamic_sample_interval/dt)
            self.snapshot_log_interval = int(self.snapshot_log_interval/dt)
            self.update_interval = int(self.update_interval/dt)
            self.time_unit = "steps" # indicate that all time variables are in units of steps

"""
-   Param change dict not implemented
"""