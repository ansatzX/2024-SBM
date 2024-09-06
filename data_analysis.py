import enum
import numpy as np
import os
from typing import List
from enum import Enum

import pickle
from matplotlib import pyplot as plt
import matplotlib
import shutil
import seaborn as sns

from traj_run import rho_ohmic
from typing import Any, List, Union
from numpy import dtype, ndarray
import scipy
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
# import pywt

def num_monotonic(p0: float, p1: float, p2: float) -> int :
    """
    filed 3
    return flag
    0: ascend
    1: decend
    2: non monotonic peak
    3: non monotonic valley
    """
    assert isinstance(p0, float)
    assert isinstance(p1, float)
    assert isinstance(p2, float)

    sorted_ascend: List[float] = sorted([p0, p1, p2])

    if sorted_ascend[0] == p2:
        return 3
    elif sorted_ascend[2] == p2:
        return 2
    elif sorted_ascend[0] == p0 and sorted_ascend[2] == p2:
        return 0
    elif sorted_ascend[0] == p2 and sorted_ascend[2] == p0:
        return 1
    
def line_monotonic_detect(data: List[float]) -> List[int]:

    if data[0] < data[1]:
        is_monotonic_results = [0]
    else: # can not be same
        is_monotonic_results = [1]

    for id in range(1, len(data)-1):
        result: int = num_monotonic(data[id-1], data[id], data[id+1])
        is_monotonic_results.append(result)

    if data[len(data)-2] < data[len(data)-1]:
        is_monotonic_results.append(0)
    else:
        is_monotonic_results.append(1)

    return is_monotonic_results

class Line_Type(Enum):
    # simply line
    One_Peak=0.1
    One_Valley=0.2
    Ascend=1.1
    Decend=1.2
    # complicated
    Oscillation=2
    Single_Minimum_And_Subsequent_Decay = 2.0
    # Oscillation_SamePeriod=2.1
    # Oscillation_PartSamePeriod=2.2
    # Oscillation_NoPeriod=2.3
    
    # Peak_WithOscillation=3.1

def line_classify(data: np.ndarray) -> int:
    """
    see def LINE TYPE
    """
    is_monotonic_results = line_monotonic_detect(data.tolist())
    # if is_monotonic_results.count(2) == 1:
    #     return Line_Type.One_Peak
    # elif is_monotonic_results.count(3) == 1:
    #     return Line_Type.One_Valley
    # elif all(is_monotonic_results) :
    #     # if oscillation, 0 1 2 atleast exists one, so if true , no 0, also no 2, Decend
    #     return Line_Type.Decend
    # elif not all(is_monotonic_results) and is_monotonic_results.count(2)==0 and is_monotonic_results.count(3) == 0:
    #     return Line_Type.Ascend
    # else:
    #     return 0
    #     if is_monotonic_results.index(2) > is_monotonic_results.index(3):
    #         starting_label = 3
    #     else:
    #         starting_label = 2
    #     count = 0
    #     period = []
    #     for id in range(len(is_monotonic_results)):
            
    #         if is_monotonic_results[id] == starting_label:
    #             if len(period) != 0:
    #                 period.append(count)
    #             count = 0
    #         count: int = count + 1
        
    #     if len(period) > 2:
    #         pass

    max_, _ = scipy.signal.find_peaks(data)
    min_, _ = scipy.signal.find_peaks(-data)

    if max_.__len__() == 1 and min_.__len__() == 0 :
        return Line_Type.One_Peak
    elif min_.__len__() == 1 and max_.__len__() == 0 :
        return Line_Type.One_Valley
    elif all(is_monotonic_results) :
        # if oscillation, 0 1 2 atleast exists one, so if true , no 0, also no 2, Decend
        return Line_Type.Decend
    elif not all(is_monotonic_results) and is_monotonic_results.count(2)==0 and is_monotonic_results.count(3) == 0:
        return Line_Type.Ascend
    else:
        if max_.__len__() == 1 and min_.__len__() == 1  and (max_[0] > min_[0]):
            return Line_Type.Single_Minimum_And_Subsequent_Decay
        elif max_.__len__() > 1 and min_.__len__() > 1 :
            return Line_Type.Oscillation
        else:
            return 0
        
def read_line(prefix_folder, line_dict) -> np.ndarray:
    infos_ = []
    for key in sorted(list(line_dict.keys())):
        step_pickle = line_dict[key]
        
        with open(os.path.join(prefix_folder, step_pickle), 'rb') as f:
            infos = pickle.load(f)
            # vn_entropy_1site/ mutual infos
        infos_.append(infos)
    return infos_


def read_omega(prefix_folder) -> np.ndarray:
    if os.path.exists(os.path.join(prefix_folder, 'sdf_wang1_omega.pickle')):
        with open(os.path.join(prefix_folder, 'sdf_wang1_omega.pickle'), 'rb') as f:
            omegas = pickle.load(f)
        # vn_entropy_1site
        return omegas
    else:
        rho_type = 0 if 'rho_type' not in os.path.basename(prefix_folder) else int(os.path.basename(prefix_folder).split('rho_type_')[1])

        s = float(os.path.basename(prefix_folder).split('_')[1][1:])
        alpha = float(os.path.basename(prefix_folder).split('_')[2][5:])

        Omega = int(os.path.basename(prefix_folder).split('_')[3][5:])
        omega_c = int(os.path.basename(prefix_folder).split('_')[5][1:])
        nmodes = int(os.path.basename(prefix_folder).split('_nmodes')[1].split('_')[0])
        # this tranlslate since 10.1103/PhysRevLett.129.120406
        s_reno, alpha_reno, omega_c_reno = translate_param(s, alpha, omega_c, Omega)
        sdf = rho_ohmic(alpha_reno, omega_c_reno, s_reno, rho_type)
        w, c2 = sdf.Wang1(nmodes)
        return w

def read_exp(prefix_folder):

    with open(os.path.join(prefix_folder, 'expectations.pickle'), 'rb') as f:
        omegas: List = pickle.load(f)

    return omegas

def get_rho_array(alpha, s, omega_c, nmodes, rho_type):
    # here should receive translated param
    sdf = rho_ohmic(alpha, omega_c, s, rho_type)
    w, c2 = sdf.Wang1(nmodes)
    rho_array = sdf._dos_Wang1(nmodes, w)
    return rho_array

def translate_param(s, alpha, omega_c, Omega):
    # 10.1103/PhysRevLett.129.120406
    s_reno = s
    alpha_reno = 4*alpha # tranlate from wang1 to PRL
    omega_c_reno  = omega_c * Omega

    return s_reno, alpha_reno, omega_c_reno

def draw_w_S(prefix_folder, dat_dict, key) -> None:
 
    s = float(os.path.basename(prefix_folder).split('_')[1][1:])
    alpha = float(os.path.basename(prefix_folder).split('_')[2][5:])

    Omega = int(os.path.basename(prefix_folder).split('_')[3][5:])
    omega_c = int(os.path.basename(prefix_folder).split('_')[5][1:])
    nmodes = int(os.path.basename(prefix_folder).split('_nmodes')[1].split('_')[0])

    rho_type = 0 if 'rho_type' not in os.path.basename(prefix_folder) else int(os.path.basename(prefix_folder).split('rho_type_')[1])

    # s_reno, alpha_reno, omega_c_reno = translate_param(s, alpha, omega_c, Omega)

    # info_ = read_line(prefix_folder, dat_dict[key])

    # omgeas = read_omega(prefix_folder)
    # omgeas_eff: ndarray[Any, dtype[Any]] = np.array([ omgeas[i] for i in range(nmodes) if f'v_{i:03}' in info_[0].keys() ] )

    
    # rho_array = get_rho_array(alpha_reno, s_reno, omega_c_reno, nmodes, rho_type)
    # rho_array_eff: ndarray[Any, dtype[Any]] = np.array([ rho_array[i] for i in range(nmodes) if f'v_{i:03}' in info_[0].keys() ] )
    omgeas_eff, rho_array_eff, modes_eff, dats = chunk_data(prefix_folder, dat_dict, key)
    # query_modes_eff = [ j for j in [ i for i in range(nmodes) if f'v_{i:03}' in modes_eff] if j in query_modes  ]

    fig_folder  = f'figs/{key}'# _nmodes{nmodes}_rho_type_{rho_type}'
    print(fig_folder)
    os.makedirs(fig_folder, exist_ok=True)

    # S_ = {}
    # for i_mode in range(omgeas_eff.shape[0]):
    #     w = omgeas_eff[i_mode]
    #     if w < omega_c * Omega:
    #         S_[i_mode] = [dat[i_mode] for dat in dats ]
    #         if i_mode in query_modes_eff:
    #             S_data = S_[i_mode]
    #             w = omgeas_eff[i_mode]
    #             plt.plot(range(0,100), S_data,'-', label=f'query_mode:{i_mode}')
    #             plt.title(f't-S*rho: {key} {rho_type}')
    #             plt.xlabel(f't')
    #             plt.xlim(0, 100)
    #             plt.ylabel(f'S')
    #             plt.legend()
    #             plt.savefig(f'{fig_folder}/t-S_rho{rho_type}_query_mode_{i_mode}.png')
 

    # uncomment this block to gen figs t-S
    # for i in range(nmodes):
    #     v_dof = f'v_{i:03}'wawd
        
    #     if v_dof in info_[0].keys():
    #         dat = []
    #         for i_step in range(100):
    #             dat.append(info_[i_step][v_dof])
    #         plt.clf()
    #         plt.plot(dat)
    #         plt.title(f't-S: {v_dof}')
    #         plt.savefig(f'{fig_folder}/t-S_rho{rho_type}_{v_dof}.png')
    #         plt.clf()





    y_lim = max([max(dat)for dat in dats])
    for i_step in range(100):
        dat = dats[i_step]
        # print(dat.shape)
        # print(omgeas_eff.shape)
        plt.plot(omgeas_eff, dat, '-') # * rho_array_eff in chunk data
        plt.title(f'w-S*rho: {key} {rho_type}')
        plt.xlabel(f'omega')
        plt.xlim(0, omgeas_eff.max())
        plt.ylabel(f'S')
        plt.ylim(0, y_lim)
        # plt.savefig(f'figs/{fig_folder}/w-S_nmodes{nmodes}_rho_type{rho_type}_{i_step:03}.png')
        # plt.clf()



def draw_t_S(prefix_folder, dat_dict, key, query_mode: int):
    """
    --input--
    prefix_folder: output folder of traj_run
    dat_key: dict from function read_line
    key: key for specified line/traj
    query_mode: the dof to draw

    --output--
    w: omega of query_mode
    freq: freq of t-S from fft
    amplitude: amplitude of t-S from fft
    phase: phase from fft
    """
    # s = 0.7
    # alpha = 0.05
    s = float(os.path.basename(prefix_folder).split('_')[1][1:])
    alpha = float(os.path.basename(prefix_folder).split('_')[2][5:])

    Omega = int(os.path.basename(prefix_folder).split('_')[3][5:])
    omega_c = int(os.path.basename(prefix_folder).split('_')[5][1:])
    nmodes = int(os.path.basename(prefix_folder).split('_nmodes')[1].split('_')[0])

    rho_type = 0 if 'rho_type' not in os.path.basename(prefix_folder) else int(os.path.basename(prefix_folder).split('rho_type_')[1])
    # pf = os.path.join(prefix_folder, f'traj_s{s:.02f}_alpha{alpha:.02f}_Omega1_omega_c10_nmodes1000_bond_dims20_td_method_0')
    # key = f's{s:.02f}-alpha{alpha:.02f}'
    omgeas_eff, rho_array_eff, modes_eff, dats = chunk_data(prefix_folder, dat_dict, key)
    if query_mode not in modes_eff:
        returntuple = ( 0, 0, 0, modes_eff)
    
    # S_ = {}
    # for i_mode in range(omgeas_eff.shape[0]):
    #     w = omgeas_eff[i_mode]
    #     if w < Omega * omega_c:
    #         S_[i_mode] = [dat[i_mode] for dat in dats ]
    w = omgeas_eff[query_mode]
    if w > Omega * omega_c:
        returntuple = ( 0, 0, 0, modes_eff)
    
    # query_index = np.where(omgeas_eff == query_mode)[0][0]
    S = [dat[query_mode] for dat in dats ]
    # w = omgeas_eff[query_index]
    plt.plot(range(0,100), S,'-', label=f'query_mode: {query_mode}')
    plt.title(f't-S*rho: {key} {rho_type}')
    plt.xlabel(f't')
    plt.xlim(0, 100)
    plt.ylabel(f'S')
    plt.legend()
    # plt.ylim(0, y_lim)
    # plt.savefig(f'figs/{key}_nmodes{nmodes}_rho_type_{rho_type}/w-S_nmodes{nmodes}_rho_type{rho_type}_{i_step:03}.png')
    # plt.clf()
    x_uniform, singnal_niform = interp_dat(np.arange(0,100), S)
    xf, yf = do_fft(x_uniform, singnal_niform)
    freq, amplitude, phase = fft_analysis(xf, yf, x_uniform.shape[0])
    return w, freq, amplitude, phase

def chunk_data(prefix_folder, dat_dict, key):
    s = float(os.path.basename(prefix_folder).split('_')[1][1:])
    alpha = float(os.path.basename(prefix_folder).split('_')[2][5:])

    Omega = int(os.path.basename(prefix_folder).split('_')[3][5:])
    omega_c = int(os.path.basename(prefix_folder).split('_')[5][1:])
    nmodes = int(os.path.basename(prefix_folder).split('_nmodes')[1].split('_')[0])

    rho_type = 0 if 'rho_type' not in os.path.basename(prefix_folder) else int(os.path.basename(prefix_folder).split('rho_type_')[1])

    s_reno, alpha_reno, omega_c_reno = translate_param(s, alpha, omega_c, Omega)

    info_ = read_line(prefix_folder, dat_dict[key])
    omgeas = read_omega(prefix_folder)


    rho_array = get_rho_array(alpha_reno, s_reno, omega_c_reno, nmodes, rho_type)
    modes_eff = [ f'v_{i:03}' for i in range(nmodes) if f'v_{i:03}' in info_[0].keys() ]
    omgeas_eff: ndarray[Any, dtype[Any]] = np.array([ omgeas[i] for i in range(nmodes) if f'v_{i:03}' in info_[0].keys() ] )
    rho_array_eff: ndarray[Any, dtype[Any]] = np.array([ rho_array[i] for i in range(nmodes) if f'v_{i:03}' in info_[0].keys() ] )

    dats = []
    for i_step in range(100):
        # S of specific mode i
        dat = [ info_[i_step][f'v_{i:03}'] for i in range(nmodes) if f'v_{i:03}' in info_[0].keys() ] 
        
        dats.append(dat*rho_array_eff)
    return omgeas_eff, rho_array_eff, modes_eff, dats


def interp_dat(x, y):
    N = x.shape[0]
    x_uniform = np.linspace(x.min(), x.max(), N)
    interpolator = interp1d(x, y, kind='cubic')
    y_uniform = interpolator(x_uniform)

    return x_uniform, y_uniform

def do_fft(x_uniform, y_uniform):
    N = x_uniform.shape[0]
    T = (x_uniform.max() - x_uniform.min()) / N  # sample Time

    yf = fft(y_uniform)
    xf = fftfreq(N, T)[:N//2]

    return xf, yf

def fft_analysis(xf, yf, N):

    fft_amp = 2.0/N *np.abs(yf[:N//2])
    indexs , _= scipy.signal.find_peaks(fft_amp)
    if len(indexs) != 0 : 
        peaks = [yf[index] for index in indexs]
        index = indexs[peaks.index(max(peaks))]
        print(peaks.index(max(peaks)), index)
        # index = indexs[0]
        freq =  xf[index]
        amplitude = 2.0/N * np.abs(yf[index])
        phase = np.angle(yf[index])
        return freq, amplitude, phase
    else:
        return 0, 0, 0
    
def func_gentor(param_a, param_b, param_c, d=None):

    def signal_func0(x, a, b, c, d):
        return a * np.exp(-b* (x**c) )+ param_a * np.sin(2.0 * np.pi * param_b * x + param_c)  + d
    def signal_func1(x, a, b, c, d):
        return a * np.exp(-b* (x**c) )  + d
    def signal_func2(x, a, b, c, d):
        a * np.sin(2.0 * np.pi * b * x + c)  + d
    if d == None:
        if param_a == 0:
            return signal_func1
        else:
            return signal_func0
    else:
        return signal_func2
