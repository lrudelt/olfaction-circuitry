from matplotlib import gridspec
from matplotlib.axes import Axes
from classes import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.cluster import KMeans
from scipy.stats.kde import gaussian_kde

def plot_dynamic_var(axs, timetrace, var, name, spline=False):
    if spline:
        from scipy.interpolate import make_interp_spline
        times_long = np.linspace(timetrace[0], timetrace[-1], 300)
        spl = make_interp_spline(timetrace, var, k=2)
        loss_smooth = spl(times_long)
        #axs.plot(timetrace, loss)
        axs.plot(times_long, loss_smooth)
    else:
        axs.plot(timetrace, var)
    axs.set_xlabel("Time (ms)")
    axs.set_ylabel("%s" % name)

def plot_vars(axs, timetrace, var1, var2, name1, name2):
    dup = axs.twinx()
    timetrace /= 1000
    axs.set_xlabel("Time (s)")
    axs.set_ylabel("%s" % name1)
    dup.set_ylabel("%s" % name2)
    p1, = axs.plot(timetrace, var1, label=name1, color='r')
    p2, = dup.plot(timetrace, var2, label=name2, color='k')
    lns = [p1, p2]
    axs.legend(handles=lns, loc='best')
    axs.set_xlim(timetrace[0], timetrace[-1])

def plot_reconstruction(axs, input, reconstruction, s):
    cols = 11
    subgrid = gridspec.GridSpecFromSubplotSpec(1, cols, subplot_spec=axs, wspace=0.05)
    input_ax = plt.subplot(subgrid[:, 0:5])
    recon_ax = plt.subplot(subgrid[:, 5:10])
    color_ax = plt.subplot(subgrid[:, 10])
    minval = min(np.amin(input), np.amin(reconstruction))
    maxval = max(np.amax(input), np.amax(reconstruction))
    inp = input[:s.n_odors]
    i = input_ax.imshow(inp, aspect="auto", vmin=minval, vmax=maxval)
    input_ax.set_xlabel("Mean PN Rate")
    input_ax.set_ylabel("Odor #")
    input_ax.set_title("Input")
    recon_ax.set_title("Reconstruction")
    rec = reconstruction[:s.n_odors]
    r = recon_ax.imshow(rec, aspect="auto", vmin=minval, vmax=maxval)
    plt.colorbar(mappable=i, cax=color_ax)

def plot_matrix(axs, matrix):
    axs.imshow(matrix, aspect='auto')
    axs.axis('off')

def plot_matrix_pdf(axs, m1, m2, name1, name2):
    dup = axs.twiny()
    e1 = m1.flatten()
    e2 = m2.flatten()
    kde1 = gaussian_kde(e1)
    kde2 = gaussian_kde(e2)
    x1 = np.linspace(np.amin(e1), np.amax(e1), 1000)
    x2 = np.linspace(np.amin(e2), np.amax(e2), 1000)
    p1, = axs.plot(x1, kde1(x1), label=name1, color='gold')
    p2, = dup.plot(x2, kde2(x2), label=name2, color='teal')
    axs.set_ylabel("PDF")
    axs.set_xlabel("%s" % name1)
    dup.set_xlabel("%s" % name2)
    lns = [p1, p2]
    axs.legend(handles=lns, loc='best')

def plot_weights(ax, F, W):
    cols = 3
    subgrid = gridspec.GridSpecFromSubplotSpec(1, cols, subplot_spec=ax)
    F_ax = plt.subplot(subgrid[:, 0])
    W_ax = plt.subplot(subgrid[:, 1])
    p_ax = plt.subplot(subgrid[:, 2])
    plot_matrix(F_ax, F)
    plot_matrix(W_ax, W)
    F_ax.set_title("Feedforward")
    W_ax.set_title("Recurrent")
    plot_matrix_pdf(p_ax, F, W, "Feedforward", "Recurrent")

def plot_odor(axs, s, repeat, odor_shape):
    assert isinstance(s, Settings)
    select_input = s.input_selector
    rates = np.swapaxes(odor_shape, 0, 1)
    shape = [select_input(rates, t, s)[0] for t in range(s.presentation_length)]
    #shape = rates.flatten()
    plot_shape = np.tile(shape, repeat)
    axs.plot(np.arange(len(plot_shape)), plot_shape)
    for r in range(repeat):
        axs.axvline(r*s.presentation_length, linestyle="--", color='k')
    axs.set_xlim(0, len(plot_shape))
    axs.axis('off')

def plot_raster(ax, spiketrace, s, nneuron=10, nodors=1):
    assert isinstance(ax, Axes)
    assert isinstance(s, Settings)
    ## Spiketrace is (ntimesteps, nneuron) with 1 (True) iff spike
    times, neuron = np.nonzero(spiketrace[:int(s.presentation_length*nodors/s.snapshot_log_interval), :nneuron])
    ax.eventplot(times[:,np.newaxis]*s.dt, orientation='horizontal', lineoffsets=neuron, colors="cornflowerblue")
    ax.set_ylabel('Neuron #')
    ax.set_xlabel('Time (ms)')
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

def plot_example(ax, input, recon, spikes, s, odor_shape, n_sample=None, n_neuron=None):
    if n_neuron is None:
        n_neuron = s.n_neuron
    if n_sample is None:
        n_sample = min(10, s.n_odors)
    odor = 1
    PN = 5
    neuron = int(PN * s.n_neuron/s.n_ORN)
    rows = odor + PN*2 + neuron
    subgrid = gridspec.GridSpecFromSubplotSpec(rows, 1, subplot_spec=ax, wspace=0.05)
    odor_ax = plt.subplot(subgrid[0:odor, :])
    inp_ax = plt.subplot(subgrid[odor:PN+odor, :])
    rec_ax = plt.subplot(subgrid[PN+odor:2*PN+odor, :])
    ras_ax = plt.subplot(subgrid[2*PN+odor:, :])
    ras_ax.set_xlim(0, s.presentation_length*n_sample*s.dt)

    plot_odor(odor_ax, s, n_sample, odor_shape)
    plot_matrix(inp_ax, np.reshape(input[:n_sample, :, :], (-1, s.n_ORN)).T)
    plot_matrix(rec_ax, np.reshape(recon[:n_sample, :, :], (-1, s.n_ORN)).T)
    plot_raster(ras_ax, spikes, s, nneuron=n_neuron, nodors=n_sample)

def plot_sparsity_fractions(ax1, ax2, rates, presentation_length, fade_frac, cutoff=None):
    ## Plot histogram of average firing rate for neurons over odor presentation
    n_neurons = rates.shape[1]
    odor_rates = np.reshape(rates, (-1, int(presentation_length*(1-fade_frac)), n_neurons))
    odor_pres_avg_rates = np.mean(odor_rates, axis=1)
    ## On first axis plot the distribution of avg firing rates
    avg_rates = odor_pres_avg_rates.flatten()
    ax1.hist(avg_rates, density=True, bins=30)
    ax1.set_xlabel("Average Firing Rates of Neurons during Odor")
    ## On the second plot the fraction of cells firing for each odor
    if cutoff is None:
        km = KMeans(n_clusters=2)
        km.fit(np.reshape(avg_rates, (-1, 1)))
        cutoff = np.mean(km.cluster_centers_)
    fraction = np.mean(np.where(odor_pres_avg_rates > cutoff, 1, 0), axis=1)
    ax2.hist(fraction, density=True, bins=20)
    ax2.set_xlabel("Fraction of Cells Firing per Odor")

def plot_training_metric(ax, hist_dict, metric_name="accuracy"):
    epochs = np.arange(len(hist_dict['%s' % metric_name]))
    ax.plot(epochs, hist_dict['%s' % metric_name])
    ax.plot(epochs, hist_dict['val_%s' % metric_name])
    ax.set_title('%s' % metric_name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel("Classifier Accuracy")
    ax.legend(['training', 'validation'], loc='best')

def plot_confusion_matrix(axs, avg_odor_prob, name):
    ## avg_odor_prob is a (nodors, nodors) array where each column is the average probability dist that the classifier outputs
    ## for the classification of a particular odor (classified over many noise instantiations)
    im = axs.imshow(avg_odor_prob, vmin=0, vmax=1)
    axs.axis('off')
    #plt.colorbar(im)
    axs.set_title('%s' % name)

def plot_learning_speed(axs, s, training_repeats, mbon_accuracy, clas_accuracy):
    axs.plot(training_repeats, mbon_accuracy, color='b', label="MBON Accuracy")
    axs.plot(training_repeats, clas_accuracy, color='g', label="Classifier Accuracy")
    axs.axhline(0.5, color='b', linestyle='--')
    axs.axhline(1/s.n_odors, color='g', linestyle='--')
    axs.set_title('Associative Learning Efficiency')
    axs.set_xlabel("# presentations of each odor")
    axs.set_ylabel("Accuracy")
    axs.legend(loc='best')
    


## MNIST functions (to make 1-D stimuli 2-D)
def plot_test_image(axs, matrix, name, nimage=16):
    m = matrix[:nimage]
    imsize = int(np.sqrt(m.shape[1]))
    img = gallery(np.reshape(m, (-1, imsize, imsize)), spacing=3)
    im = axs.imshow(img, cmap='Greys', vmin=0, vmax=1)
    axs.axis('off')
    axs.set_title(r'$%s$' % name)

def plot_receptor_field(axs, matrix, name):
    field_min = -0.1
    field_max = 1.0
    fields = np.swapaxes(matrix ,0,1)
    imsize = int(np.sqrt(fields.shape[1]))
    img = gallery(np.reshape(fields, (-1, imsize, imsize)), spacing=3, spacing_value=field_min)
    im = axs.imshow(img, cmap='Greys', vmin=field_min, vmax=field_max)
    axs.axis('off')
    axs.set_title(r'$%s$' % name)

def gallery(array, ncols=-1, spacing=2, spacing_value=None):
    if ncols == -1:
        ncols = np.math.ceil(np.sqrt(array.shape[0]))
    nrows = np.math.ceil(array.shape[0]/float(ncols))
    cell_w = array.shape[2]
    cell_h = array.shape[1]
    if spacing_value is None:
        spacing_value = np.min(array)
    result = np.ones(((cell_h+spacing)*nrows + spacing, (cell_w+spacing)*ncols + spacing), dtype=array.dtype) * spacing_value
    s = spacing
    for i in range(0, nrows):
        for j in range(0, ncols):
            if i*ncols+j < array.shape[0]:
                result[i*(cell_h+s)+s:(i+1)*(cell_h+s), j*(cell_w+s)+s:(j+1)*(cell_w+s)] = array[i*ncols+j]
    return result

def plot_mnist(log_file):
    ## Open log file
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

    dyn = 2
    neuron = 5

    grid = gridspec.GridSpec(9, 4, hspace=2.5)
    dec = plt.subplot(grid[0:dyn, :])
    row = dyn
    mat = plt.subplot(grid[row:row+neuron, :])
    Fax = plt.subplot(grid[4, :])
    tru = plt.subplot(grid[5:7, 0])
    rec = plt.subplot(grid[5:7, 1])
    Fnn = plt.subplot(grid[5:7, 2])
    Dnn = plt.subplot(grid[5:7, 3])
    ras = plt.subplot(grid[7:9, :])
    #odo = plt.subplot(grid[8:9, :])

    plot_dynamic_var(dec, timetrace, decoder_loss, "Decoder Loss")
    plot_weights(mat, F, W)
    # plot_test_image(tru, input, name="Image")
    # plot_test_image(rec, recon, name="Reconstruction")
    # plot_receptor_field(Fnn, F.T, name='F')
    # plot_receptor_field(Dnn, D, name='D')
    plot_raster(ras, spikes, "Spiketrain", s)
    #plot_odor(odo, inputs, "Odor", s)
    plt.savefig("%s/Overview_%s" % (s.folder_name, s.project_name))
    plt.close()

def plot_pars(ax, s):
    assert isinstance(s, Settings)
    ax.axis("off")
    ax.text(1, 0.5, 'Dataset: %s\n # Odors: %s\n # ORN: %s' % (s.odor_file[s.odor_file.rfind("/"):], s.n_odors, s.n_ORN) , style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10}, fontsize=8)
    ax.text(0.5, 0.5, '# Neurons: %s\n Mean Firing Rate: %s\n # Unique Train, Test Odors: %s, %s' % (s.n_neuron, s.rho, s.n_train_repeat, s.n_test_repeat) , style='italic',
        bbox={'facecolor': 'orange', 'alpha': 0.5, 'pad': 10}, fontsize=8)
    ax.text(0, 0.5, 'Input Selector: %s\n Adaptation: %s\n Presentation Length, Fade Frac: %s, %s\n dt (ms): %s' % (s.input_selector.__name__, s.input_adaptation, s.presentation_length*s.dt, s.fade_fraction, s.dt) , style='italic',
        bbox={'facecolor': 'lightsteelblue', 'alpha': 0.5, 'pad': 10}, fontsize=8)
    
    ## Parameters to add to plot:
    ## 1. Input data: dataset, n_odor, n_ORN
    ## 2. Network: n_neuron, rho, train size, test size, plasticity, learning rates
    ## 3. Input dynamics: shape of profile, presentation length, stim_on_frac


def plot_odor_full_alt(log_file, rates, hist, confusion, learn_dict):
    ## Open log file
    with gzip.open(log_file, 'r') as f:
        log = pickle.load(f)
    assert isinstance(log, Log)
    final_time = max(log.snapshots.keys())
    if final_time > log.dynamic_log["t"][-1]:
        dyn_log_idx = len(log.dynamic_log["t"])
    else:
        dyn_log_idx = np.where(log.dynamic_log["t"] == final_time)[0][0]
    
    final_snap = log.snapshots[max(log.snapshots.keys())]
    F, D, W = final_snap["feedforward_weights"], final_snap["decoder"], final_snap["recurrent_weights"]
    input, recon = final_snap["inputs"], final_snap["reconstruction_means"]
    spikes = final_snap["spikes"]
    s = final_snap["s"]
    timetrace = log.dynamic_log["t"][:dyn_log_idx]*s.dt
    decoder_loss = log.dynamic_log["test_decoder_loss"][:dyn_log_idx]
    firing_rate = log.dynamic_log["avg_firing_rate"][:dyn_log_idx]
    print("Time:", max(log.snapshots.keys()), log.dynamic_log["t"][dyn_log_idx-1])

    fig = plt.figure(figsize=(25, 25))
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

def plot_odor_full(log_file, rates, hist, confusion, acc_dict, odor_shape):
    ## Open log file
    with gzip.open(log_file, 'r') as f:
        log = pickle.load(f)
    assert isinstance(log, Log)
    
    ## Make sure that logging variables and snapshot variables are taken from about the same time
    final_time = max(log.snapshots.keys())
    if final_time > log.dynamic_log["t"][-1]:
        dyn_log_idx = len(log.dynamic_log["t"])
    else:
        dyn_log_idx = np.where(log.dynamic_log["t"] == final_time)[0][0]
    final_snap = log.snapshots[max(log.snapshots.keys())]

    ## Pull snapshot variables
    F, D, W = final_snap["feedforward_weights"], final_snap["decoder"], final_snap["recurrent_weights"]
    inputs, inp_means, recon, recon_full = final_snap["inputs"], final_snap["input_means"], final_snap["reconstruction_means"], final_snap["reconstructions"]
    spikes = final_snap["spikes"]
    s = final_snap["s"]
    assert isinstance(s, Settings)
    ## Pull logging variables
    timetrace = log.dynamic_log["t"][:dyn_log_idx]*s.dt
    decoder_loss = log.dynamic_log["test_decoder_loss"][:dyn_log_idx]
    firing_rate = log.dynamic_log["avg_firing_rate"][:dyn_log_idx]
    print("Snapshot time (s):", max(log.snapshots.keys())*s.dt/1000, "\nMax Log Time (s):", log.dynamic_log["t"][dyn_log_idx-1]*s.dt/1000)

    fig = plt.figure(figsize=(25, 25))
    dyn = 4
    neuron = 5
    #plt.tight_layout()
    row = 0
    grid = gridspec.GridSpec(neuron*4 + dyn*3, 8, hspace=6.5, wspace=1)
    var = plt.subplot(grid[row:row+dyn, :])
    row += dyn
    mat = plt.subplot(grid[row:row+neuron, :])
    row += neuron
    rec = plt.subplot(grid[row:row+neuron, :])
    row += neuron
    exa = plt.subplot(grid[row:row+neuron*2, :])
    row += neuron*2
    sp1 = plt.subplot(grid[row:row+dyn, 0:2])
    sp2 = plt.subplot(grid[row:row+dyn, 2:4])
    acc = plt.subplot(grid[row:row+dyn, 4:6])
    con = plt.subplot(grid[row:row+dyn, 6:8])
    row += dyn
    lea = plt.subplot(grid[row:, 0:3])
    par = plt.subplot(grid[row:, 3:])

    plot_vars(var, timetrace, decoder_loss, firing_rate, "Decoder Loss", "Mean Firing Rate (spikes/neuron/ms)")
    plot_weights(mat, F, W)
    plot_reconstruction(rec, inp_means, recon, s)
    plot_example(exa, inputs, recon_full, spikes, s, odor_shape)
    if rates is not None:
        plot_sparsity_fractions(sp1, sp2, rates, s.presentation_length, s.fade_fraction)
    plot_confusion_matrix(con, confusion, "Confusion Matrix")
    plot_training_metric(acc, hist)
    plot_learning_speed(lea, s, acc_dict["Repeats"], acc_dict["MBON"], acc_dict["Classifier"])
    plot_pars(par, s)
    plt.savefig("%s/%s" % (s.folder_name, s.project_name))
    plt.close()