import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from classes import *

def randomify_feedforward(F, shuffle=True, seed=0):
    np.random.seed(seed=seed)
    flat = F.flatten()
    if shuffle:
        np.random.shuffle(flat)
        output = np.reshape(flat, F.shape)
    else:
        ## Sampling version of shuffle
        ## TODO: Would also want to implement indegree constancy but need pruning first
        flat = F.flatten()
        bandwidths = 10 ** np.linspace(-1, 1, 100)
        kde = GridSearchCV(KernelDensity(kernel='gaussian'),
                    {'bandwidth': bandwidths},
                    cv=LeaveOneOut(len(flat))).best_estimator_
        kde.fit(flat)
        output = np.reshape(kde.sample(n_samples=len(flat)), F.shape)
    return output

def get_odor_rate(x, t, s):
    assert isinstance(s, Settings)
    if s.time_unit == "ms":
        l = int(s.presentation_length/s.dt)
    elif s.time_unit == "steps":  
        l = s.presentation_length  
    odor = int(t/l)
    timestep = t % l
    return x[odor, timestep, :]

def odor_flat_linchange(x, t, s):
    assert isinstance(s, Settings)
    l = s.presentation_length
    peak_t = s.input_tau_inc
    fade_frac = s.fade_fraction 
    odor = x[int(t/l), peak_t, :]
    frac = (t % l)/l
    if frac > 1 - fade_frac:
        next_odor = x[min(int(t/l)+1, len(x)-1), peak_t, :]
        return next_odor*(1-(1-frac)/fade_frac) + odor*(1-frac)/fade_frac
    else:
        return odor