import numpy as np
import matplotlib as mpl
import math
import scipy.stats as stats
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from dataclasses import dataclass
from typing      import Tuple
from typing      import TypeVar
from typing      import Callable
from collections import namedtuple
from scipy.optimize          import OptimizeWarning
import warnings
from matplotlib.ticker import AutoMinorLocator

from invisible_cities.core.core_functions  import shift_to_bin_centers
from invisible_cities.core                 import fit_functions as fitf
from invisible_cities.evm  .ic_containers import FitFunction
from   invisible_cities.core.core_functions    import in_range
from   invisible_cities.core .stat_functions import poisson_sigma

Measurement = namedtuple('Measurement', 'value uncertainty')
Number = TypeVar('Number', None, int, float)
NN = np.nan

@dataclass
class HistoPar:
    var    : np.array
    nbins  : int
    range  : Tuple[float]

@dataclass
class GaussPar:
    mu    : Measurement
    std   : Measurement
    amp   : Measurement

@dataclass
class ProfilePar:
    x  : np.array
    y  : np.array
    xu : np.array
    yu : np.array

@dataclass
class FitPar(ProfilePar):
    f     : FitFunction

@dataclass
class FitResult:
    par   : np.array
    err   : np.array
    chi2  : float
    valid : bool

@dataclass
class FitCollection:
    fp   : FitPar
    hp   : HistoPar
    fr   : FitResult

def gaussian_parameters(x : np.array, range : Tuple[Number], bin_size : float = 1)->GaussPar:
    """
    Return the parameters defining a Gaussian
    g = N * exp(x - mu)**2 / (2 * std**2)
    where N is the normalization: N = 1 / (sqrt(2 * np.pi) * std)
    The parameters returned are the mean (mu), standard deviation (std)
    and the amplitude (inverse of N).
    """
    mu, std = mean_and_std(x, range)
    ff     = np.sqrt(2 * np.pi) * std

    amp     = len(x) * bin_size / ff

    sel  = in_range(x, *range)
    N = len(x[sel])              # number of samples in range
    mu_u  = std / np.sqrt(N)
    std_u = std / np.sqrt(2 * (N -1))
    amp_u = np.sqrt(2 * np.pi) * std_u

    return GaussPar(mu  = Measurement(mu, mu_u),
                    std = Measurement(std, std_u),
                    amp = Measurement(amp, amp_u))

def gaussian_fit(x       : np.array,
                 y       : np.array,
                 seed    : GaussPar,
                 n_sigma : int)  ->Tuple[FitPar, FitResult]:
    """Gaussian fit to x,y variables, with fit range defined by n_sigma"""

    mu  = seed.mu.value
    std = seed.std.value
    amp = seed.amp.value
    fit_range = mu - n_sigma * std, mu + n_sigma * std

    x, y      = x[in_range(x, *fit_range)], y[in_range(x, *fit_range)]
    yu        = poisson_sigma(y)
    fseed     = (amp, mu, std)

    par, err = par_and_err_from_seed(seed)
    fr = FitResult(par = par,
                   err = err,
                   chi2 = NN,
                   valid = False)
    fp = None

    with warnings.catch_warnings():
        warnings.filterwarnings('error')  # in order to handle fit failures here
        try:
            fp, fr = gfit(x, y, yu, fseed)
        except RuntimeWarning:   # this is the most usual failure, and usually solved trying fitx
                                 # with a different seed
            print(f' fit failed for seed  = {seed}, due to RunTimeWarning, retry fit ')
            fseed = (10*fseed[0], fseed[1], fseed[2] )
            try:
                fp, fr = gfit(x, y, yu, fseed)
            except RuntimeWarning: #  Give up on second failure
                print(f' fit failed for seed  = {seed}, due to RunTimeWarning, give up ')
        except OptimizeWarning:
            print(f' OptimizeWarning was raised for seed  = {seed} due to OptimizeWarning')
        except RuntimeError:
            print(f' fit failed for seed  = {seed}  due to RunTimeError')
        except TypeError:
            print(f' fit failed for seed  = {seed}  due to TypeError')

    return fp, fr

def fit_energy(e : np.array,
               nbins   : int,
               range   : Tuple[float],
               n_sigma : float = 3.0)->FitCollection:
    """
    Takes an "energy vector" (e.g, 1d array), with number of bins enbins and range erange, then:
        1. Computes the histogram of e with enbins in erange. This returns an array of bin
        edges (b), and bin contents (y). The array (b) is shifted to bin centers (x)
        2. The arrays x and y are fitted to a gaussian, in a range given by an interval
        arround the estimation of the maximum of the gaussian. The interval size is estimated
        by multiplying n_sigma by the estimation of the gaussian std.
    The result of the fit is a fit collection, that includes a FitPar and a HistoPar objects
    needed for printing and plotting the fit result.
       """

    y, b = np.histogram(e, bins= nbins, range=range)
    x = shift_to_bin_centers(b)
    bin_size = (range[1] - range[0]) / nbins
    seed = gaussian_parameters(e, range, bin_size)

    fp, fr = gaussian_fit(x, y, seed, n_sigma)

    hp = HistoPar(var      = e,
                  nbins    = nbins,
                  range    = range)

    return FitCollection(fp = fp, hp = hp, fr = fr)

def mean_and_std(x : np.array, range_ : Tuple[Number, Number])->Tuple[Number, Number]:
    """Computes mean and std for an array within a range: takes into account nans"""

    mu  = NN
    std = NN

    if all(np.isnan(x)):  # all elements are nan
        mu  = NN
        std = NN
    else:
        x_nonnan = x[np.isfinite(x)]
        y = x_nonnan[in_range(x_nonnan, *range_)]
        if len(y) == 0:
            warnings.warn(f'warning, empty slice of x = {x} in range = {range_}')
            mu  = NN
            std = NN
        else:
            mu  = np.mean(y)
            std = np.std(y)

    return mu, std

def par_and_err_from_seed(seed : GaussPar) ->Tuple[np.array, np.array]:
    par = np.zeros(3)
    err = np.zeros(3)
    par[0] = seed.amp.value
    par[1] = seed.mu.value
    par[2] = seed.std.value
    err[0] = seed.amp.uncertainty
    err[1] = seed.mu.uncertainty
    err[2] = seed.std.uncertainty
    return par, err

def gfit(x     : np.array,
         y     : np.array,
         yu    : np.array,
         fseed : Tuple[float, float, float]) ->Tuple[FitPar, FitResult]:

    f     = fitf.fit(fitf.gauss, x, y, fseed, sigma=yu)
    c2    = chi2(f, x, y, yu)
    par  = np.array(f.values)
    err  = np.array(f.errors)
    xu   = np.diff(x) * 0.5

    fr = FitResult(par = par,
                   err = err,
                   chi2 = c2,
                   valid = True)
    fp = FitPar(x  = x, y  = y, xu = xu, yu = yu, f  = f.fn)

    return fp, fr

def chi2f(f   : Callable,
          nfp : int,        # number of function parameters
          x   : np.array,
          y   : np.array,
          yu  : np.array)->float:
    """
    Computes the chi2 of a function f applied over array x and compared
    with array y with error yu
    """

    assert len(x) == len(y) == len(yu)
    fitx  = f(x)
    chi2_ = np.sum(((y - fitx) / yu)**2)

    if len(x) > nfp:
        return chi2_/(len(x)-nfp)
    else:
        warnings.warn('nof = 0 in chi2 calculation, return chi2 = {chi2_}', UserWarning)
        return chi2_



def chi2(f : FitFunction,
         x : np.array,
         y : np.array,
         sy: np.array)->float:
    """
    Computes the chi2 of a function f applied over array x and compared
    with array y with error yu. The object f is of type FitFunction.
    """
    return chi2f(f.fn, len(f.values), x, y, sy)

def plot_fit_energy(fc : FitCollection, fit_range):

    if fc.fr.valid:
        par  = fc.fr.par
        x    = fc.hp.var
        r    = 2.35 * 100 *  par[2] / par[1]
        entries  =  f'Entries = {len(x)}'
        mean     =  r'$\mu$ = {:7.2f}'.format(par[1])
        sigma    =  r'$\sigma$ = {:7.2f}'.format(par[2])
        rx       =  r'Eres (FWHM) % = {:7.2f}'.format(r)
        chi2     =  r'$\chi^2$ = {:7.2f}'.format(fc.fr.chi2)
        stat     =  f'{entries}\n{mean}\n{sigma}\n{chi2}\n{rx}'

        f, ax = plt.subplots(1,1)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        ax.tick_params(axis='both', direction='in', reset=True, labelsize=15, which='both')
        ax.tick_params(which='major', length=7)
        ax.tick_params(which='minor', length=4)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        _, _, _   = plt.hist(fc.hp.var,
                             bins = fc.hp.nbins,
                             range=fit_range,
                             histtype='step',
                             edgecolor='black',
                             linewidth=1.5,
                             label=stat)

        plt.plot(fc.fp.x, fc.fp.f(fc.fp.x), "r-", lw=4)
        anchored_text = AnchoredText(stat, loc=2)
        ax.add_artist(anchored_text)
    else:
        warnings.warn(f' fit did not succeed, cannot plot ', UserWarning)


def print_fit_energy(fc : FitCollection):

    par  = fc.fr.par
    err  = fc.fr.err
    try:
        r  = 2.35 * 100 *  par[2] / par[1]
        fe = np.sqrt(41 / 2458) * r
        print(f'  Fit was valid = {fc.fr.valid}')
        print(f' Emu       = {par[1]} +-{err[1]} ')
        print(f' E sigma   = {par[2]} +-{err[2]} ')
        print(f' chi2    = {fc.fr.chi2} ')

        print(f' Eres(FWHM)     (%) ={r}')
        print(f' Eres (FWHM) Qbb (%) ={fe} ')
    except ZeroDivisionError:
        warnings.warn(f' mu  = {par[1]} ', UserWarning)

def get_fit_params(fc: FitCollection):

    par = fc.fr.par
    eres = 2.35 * 100 * par[2] / par[1]
    fwhm = 2.35 * par[2]
    mean = par[1]

    err = fc.fr.err
    mean_err = err[1]
    fwhm_err = err[2]
    eres_err = (2.35*100/mean)*np.sqrt(fwhm_err**2. + (fwhm**2. / mean**2.)*mean_err**2.)

    chi2 = fc.fr.chi2

    return eres, fwhm, mean, eres_err, fwhm_err, mean_err, chi2

def Gaussian(mu, sigma, this_range=None):

    if this_range == None:
        this_range = (mu - 3*sigma, mu + 3*sigma)

    x = np.linspace(this_range[0], this_range[1], 100)
    y = stats.norm.pdf(x, mu, sigma)
    return x, y

def plot_fit_energy2(fc : FitCollection, data, plot_range):

    if fc.fr.valid:
        par  = fc.fr.par
        x    = fc.hp.var
        r    = 2.35 * 100 *  par[2] / par[1]
        entries  =  f'Entries = {len(x)}'
        mean     =  r'$\mu$ = {:7.2f}'.format(par[1])
        sigma    =  r'$\sigma$ = {:7.2f}'.format(par[2])
        rx       =  r'Eres (FWHM) % = {:7.2f}'.format(r)
        chi2     =  r'$\chi^2$ = {:7.2f}'.format(fc.fr.chi2)
        stat     =  f'{entries}\n{mean}\n{sigma}\n{chi2}\n{rx}'

        f, ax = plt.subplots(1,1)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        ax.tick_params(axis='both', direction='in', reset=True, labelsize=15, which='both')
        ax.tick_params(which='major', length=7)
        ax.tick_params(which='minor', length=4)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        _, _, _   = plt.hist(data,
                             bins = fc.hp.nbins,
                             range=plot_range,
                             histtype='step',
                             edgecolor='black',
                             linewidth=1.5,
                             label=stat,
                             density=True)

        x, y = Gaussian(par[1], par[2], plot_range)
        print('data', data)
        print('plot_range', plot_range)
        print('data in range', data[data > plot_range[0]])
        amp = data.count() #len(fc.hp.var) #sum(fc.fp.f(fc.fp.x))
        #y = fc.fp.f(x)
        print('y', y)
        print('amp', amp)
        print('max data', data.max())
        print('original y', fc.fp.f(fc.fp.x))
        print('A', np.sqrt(2 * np.pi)/par[2])
        """
        y, b = np.histogram(data, bins= 1000, range=[np.min(data), np.max(data)])
        x = shift_to_bin_centers(b)
        peak = x[np.argmax(y)]
        fit_range = (peak, peak + np.std(data)/4.) # only fit left half to avoid the tails

        plot_data = data[data > plot_range[0]]
        plot_data = plot_data[plot_data < plot_range[1]]
        fit_data = data[data > fit_range[0]]
        fit_data = fit_data[fit_data < fit_range[1]]
        print('Area plotted', plot_data.count())
        print('Area fitted', fit_data.count())
        print('total area', data.count())
        print('Area curve fit', par[0]*np.sqrt(2*np.pi)/par[2])

        fit_x = fc.fp.x
        fit_y = fc.fp.f(fc.fp.x)
        fit_A = len(fc.fp.x)
        """
        plt.plot(x, y, "r-", lw=4)
        anchored_text = AnchoredText(stat, loc=2)
        ax.add_artist(anchored_text)
    else:
        warnings.warn(f' fit did not succeed, cannot plot ', UserWarning)