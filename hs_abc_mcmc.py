from optparse import OptionParser
import numpy as np
from random import random as rm
import Queue as queue
import multiprocessing as mp
from ctypes import cdll
import subprocess
from subprocess import Popen, PIPE
from operator import add, sub
import random
from scipy.stats import beta
import math
from hpd import hpd_grid


USAGE= """Usage: %prog [options]"""
OPT_DEFAULTS={'infile':'-'}
DESCRIPTION="""Program description: """
EPILOG="""Requirements:"""

def get_options(defaults, usage, description='',epilog=''):
    """Get options, print usage text."""
    parser=OptionParser(usage=usage,description=description,epilog=epilog)
    parser.add_option("-i","--infile",action="store",dest="infile",type="string",
                      default=defaults.get('infile'),
                      help='Name of input gene')
    parser.add_option("-r","--runs",action="store",dest="runs",type="int",
                      default=defaults.get('runs'),
                      help='Number of runs for simulation')
    parser.add_option("-m","--mutation-rate",action="store",dest="mut",type="float",
                      default=defaults.get('mut'),
                      help='Mutation rate (float)')
    parser.add_option("-o","--obs-freq",action="store",dest="obs",type="float",
                      default=defaults.get('obs'),
                      help='Observed mutation frequnecy')

    (options,args)=parser.parse_args()

    return (options, args)

def run_simulation(runs, sel, dom, mutU):
    cmd = ["/Users/zachfuller/mut_population_sims/mut_uncert_sim","%i"%runs,"%s"%sel,"%s"%dom,"%s"%mutU]
    result = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    output = result.stdout.readlines()

    out = [float(x) for x in output[0].split()[5:]]
    freq = ((out[0]*2) + out[1])/(sum(out[:-1])*2)
    out.extend([sel, dom])

    pop_samp = np.random.binomial(113771, freq, 1)
    pop_freq = float(pop_samp)/113771

    mut_u = float(output[0].split()[-1].strip("\n"))
    #print out
    return pop_freq, out, mut_u

def lognuniform(low=1e-6, high=1, size=None, base=np.exp(1)):
    return np.power(base, np.random.uniform(np.log10(low), np.log10(high), size))

def clamp(n, minn=1e-4, maxn=1-(1e-4)):
    return max(min(maxn, n), minn)

def kde_scipy(x, obs, bandwidth=0.3, **kwargs):
    """Kernel Density Estimation with Scipy"""
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    tol = 1e-05
    try:
        kde = gaussian_kde(x, bw_method=bandwidth, **kwargs)
        return kde.integrate_box(obs-tol,obs+tol)
    except:
        return 0.0

def get_beta_params(exp_u, mutU):
    mut_samp = mutU/exp_u
    alpha = mut_samp * 113770
    beta_p = 113770 + 1
    return alpha+1, beta_p

def main():
    (options,args)=get_options(OPT_DEFAULTS, USAGE, DESCRIPTION, EPILOG)
    infile = options.infile
    runs = options.runs
    mutU = options.mut
    obs = options.obs
    if obs < 0:
        print infile, "NA", "NA","NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"
    else:
        ops = (add, sub)
        init_hs = mutU/obs
        if init_hs > 1: init_hs = .9
        alpha, beta_p = get_beta_params(init_hs, mutU)
        init_hs = clamp(mutU/np.random.beta(a=alpha,b=beta_p))
        init_s, init_h = (init_hs)**.5, (init_hs)**.5
        s, h = init_s, init_h
        hs = init_hs
        #print s, h
        init_freq = run_simulation(1, s, h, mutU)

        #init_freq = run_simulation(1, init_s, init_h, mutU)
        accept = 0
        curr_freq = init_freq[0]
        samp_size = 113770

        pi_x = beta.pdf(curr_freq, a=samp_size*obs + 1, b=samp_size-(samp_size*obs)+1)
        #pi_x = min(1-pi_x, pi_x)*2
        hs_L = []

        for run in xrange(runs):

            prop_param_s, prop_param_h = 50, 50
            prop_s  = clamp(np.random.beta(a=prop_param_s,b=(prop_param_s-(s*prop_param_s))/s))
            prop_h = clamp(np.random.beta(a=prop_param_h,b=(prop_param_h-(h*prop_param_h))/h))

            sim_result = run_simulation(1, prop_s, prop_h, mutU)
            s_xy = beta.pdf(prop_s, a=prop_param_s, b=(prop_param_s-(s*prop_param_s))/s)
            h_xy = beta.pdf(prop_h, a=prop_param_h, b=(prop_param_h-(h*prop_param_h))/h)
            s_yx = beta.pdf(s, a=prop_param_s, b=(prop_param_s-(prop_s*prop_param_s))/prop_s)
            h_yx = beta.pdf(h, a=prop_param_h, b=(prop_param_h-(prop_h*prop_param_h))/prop_h)
           
            pi_x = beta.pdf(curr_freq, a=samp_size*obs + 1, b=samp_size-(samp_size*obs)+1)
            pi_y = beta.pdf(sim_result[0], a=samp_size*obs + 1, b=samp_size-(samp_size*obs)+1)

            Q_xy = s_xy * h_xy
            Q_yx = s_yx * h_yx
            #Q_xy = hs_xy
            #Q_yx = hs_yx
            if pi_x*Q_xy > 0:
                mh_r = float(pi_y*Q_yx)/(pi_x*Q_xy)
            else:
                mh_r = 1
            if math.isnan(mh_r) == True: mh_r = 1
            ratio = min(1, mh_r)
            # #print pi_y, Q_yx, pi_x, Q_xy
            prob = rm()
            #print prob, ratio
            if prob <= ratio:
                s, h = prop_s, prop_h
                hs = s*h
                #hs = prop_hs
                accept += 1
                curr_freq = sim_result[0]
                
            #print s, h, hs, sim_result[0], obs, prob, ratio, mh_r, prop_s, prop_h, accept, Q_xy, Q_yx, pi_x, pi_y, sim_result[-1]
            hs_L.append(hs)
        hpd_mu, x_mu, y_mu, modes_mu = hpd_grid(hs_L, roundto=6)
        log_hpd_mu, log_x_mu, log_y_mu, log_modes_mu = hpd_grid(np.log10(hs_L), roundto=6)
        print infile, obs, mutU, np.mean(hs_L), np.percentile(hs_L,2.5), np.percentile(hs_L,97.5), np.median(hs_L), modes_mu[0], hpd_mu[0][0], hpd_mu[0][1], log_modes_mu[0], log_hpd_mu[0][0], log_hpd_mu[0][1]

if __name__ == '__main__':
    main()
