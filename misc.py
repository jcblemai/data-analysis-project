import numpy as np
import pandas as pd
import datetime

def upscale_to_dept(g, cases):
    dept_cases = pd.DataFrame(0, index=cases.index, columns=np.arange(0,10))
    for ws in range(g.nnodes):
        for dept in np.nonzero(g.ws_adm1[ws,:])[0]:
            dept_cases.loc[:,dept] += g.ws_adm1[ws,dept] *cases[:][ws]
            
    return dept_cases

def upscale_to_dept2(g, cases):
    dept_cases = pd.DataFrame(0, index=cases.index, columns=g.adm1_name)
    for ind,dept in enumerate(g.adm1_name):
        tot_ws = sum(g.ws_adm1[:,ind])
        for ws in np.nonzero(g.ws_adm1[:,ind])[0]:
             dept_cases.loc[:,dept] += g.ws_adm1[ws,ind] * cases[:][ws]
    return dept_cases


def upscale_with_mean(geo, data):
    dept_data = pd.DataFrame(0, index=data.index, columns=geo.adm1_name)
    for ind,dept in enumerate(geo.adm1_name):
        tot_ws = sum(geo.ws_adm1[:,ind])
        for ws in np.nonzero(geo.ws_adm1[:,ind])[0]:
             dept_data.loc[:,dept] += geo.ws_adm1[ws,ind] * data[:][ws]/tot_ws
    return dept_data

def date_mat2py(matlab_datenum):
    return datetime.datetime.fromordinal(int(matlab_datenum)) + datetime.timedelta(days=matlab_datenum%1) - datetime.timedelta(days = 366)

def timeseries_mat2py(data_mat, date_mat, columns):
    dates = []
    for ind, d in enumerate(date_mat):
        dates.append(date_mat2py(float(d)))

    return pd.DataFrame(data_mat.T, index=dates, columns = columns)

def downscale_to_ws(g, cases):
    """
        TODO: Take into account population
    """
    ws_cases = pd.DataFrame(0, index=cases.index, columns=np.arange(0,g.nnodes))
    for ind,dept in enumerate(g.adm1_name):
        tot_ws = sum(g.ws_adm1[:,ind])
        for ws in np.nonzero(g.ws_adm1[:,ind])[0]:
            ws_cases.loc[:,ws] += g.ws_adm1[ws,ind] * cases[:][dept]/tot_ws
    return ws_cases


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
