import scipy.io as sio
import numpy as np
import pandas as pd
import datetime, time
import numpy.linalg as lia
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from sirb import SIRB
from misc import *

class SIRB_Setup:

    def model_ws(self):
        """ 
            Create a Setup for a model on watershed
        """
        
        datapath = 'data/'
        
        self.dt = 1
        
        # Model timeline
        self.t1i = datetime.date(2010, 10, 24)
        self.t1f = datetime.date(2010, 10, 22)     # to force
        self.t2f = datetime.date(2015, 1, 1)       #final time of forecast
       
        self.p = SIRB_Parameters.from_list([0.901111670421407,
                                            0.0340324430375198,
                                            0.274589893352084,
                                            58.9016534494006,
                                            0.619901160642942,
                                            1.01068967295270,
                                            0.0538198449861880,
                                            0.0292689838663739,
                                            0.227960466652046,
                                            0.0695082468030896])   # Check OK

        
        # Read geography data
        self.geo = Geography(datapath + 'geodata/geodata.mat')
        self.geo.ws_list = np.load(datapath + 'geodata/ws_list.npy')
        self.geo.ws_grid = np.load(datapath + 'geodata/ws_grid_downscaled.npy')
        
        # Read rainfall data
        rainfall_raw = sio.loadmat(datapath + 'weather/prec_TRMM_GPM_daily_ws.mat')        
        self.rainfall = timeseries_mat2py(rainfall_raw['R_WS_day'], rainfall_raw['date_list'], np.arange(self.geo.nnodes)).fillna(0)  # Check OK
        
        # Read cases data
        cases_raw = sio.loadmat(datapath + 'cases/casedata_10Jan2017.mat') 
        cases = timeseries_mat2py(cases_raw['cases_week'], cases_raw['date_cases_week'], self.geo.adm1_name).resample('W-SAT').sum()
        self.cases_dept = cases.fillna(0) # because there are no cases on 2016-12-10
        self.cases_ws = downscale_to_ws(self.geo, self.cases_dept)
        
        self.cases = self.cases_ws
        
        fluxes=np.dot(np.exp(-self.geo.dist_road/self.p.D), np.diag(self.geo.popnodes)) 
        np.fill_diagonal(fluxes,0)
        self.fluxes = fluxes/np.matlib.repmat(np.sum(fluxes,axis=1),self.geo.nnodes,1) # check NEARLY OK
        self.fluxes = sio.loadmat(datapath + 'fluxes.mat') 
        self.fluxes = self.fluxes['fluxes']
        
        self.HPtH = lia.multi_dot([np.diag(1./self.geo.popnodes), self.geo.AD.T.todense(), np.diag(self.geo.popnodes)]) # check NEARLY OK
        self.HPtH = sio.loadmat(datapath + 'HPtP.mat') 
        self.HPtH = self.HPtH['HPtH'].todense()
        
        self.i = SIRB_ind(self.geo.nnodes)
        
        self.y0 = self.build_ic()
        
        return self
    
    
    def model_dept(self):
        self.model_ws()

        self.rainfall = upscale_with_mean(self.geo, self.rainfall)
        self.cases = self.cases_dept
        self.fluxes = self.geo.ws_adm1.T @ self.fluxes @ self.geo.ws_adm1
        np.fill_diagonal(self.fluxes, 0)
        row_sums = self.fluxes.sum(axis=1)
        self.fluxes = self.fluxes / row_sums[:, np.newaxis]

        self.HPtH = np.zeros((10,10))
        old_i = self.i
        self.i = SIRB_ind(10)
        
        
        self.geo.nnodes = 10
        self.geo.popnodes = self.geo.popnodes @ self.geo.ws_adm1
        self.H = np.sum(self.geo.popnodes)
        
        y0 = np.zeros((5 * self.geo.nnodes))
        y0[self.i.S] = self.y0[old_i.S] @ self.geo.ws_adm1
        y0[self.i.I] = self.y0[old_i.I] @ self.geo.ws_adm1
        y0[self.i.R] = self.y0[old_i.R] @ self.geo.ws_adm1
        y0[self.i.B] = self.y0[old_i.B] @ self.geo.ws_adm1
        y0[self.i.C] = self.y0[old_i.C] @ self.geo.ws_adm1
        
        
        self.y0 = y0
        # No transport
        self.p.l = 0
        
        self.p.muB_wsd = np.full((10, ), 0.029268983866374)
        
        
        return self
    
    def build_ic_dept(self):
        ic_pts = [150-1, 109-1, 119-1, 118-1]
        
        I0 = np.insert(1100*self.geo.popnodes.take(ic_pts[1:])/np.sum(self.geo.popnodes.take(ic_pts[1:])),0,1000)/(self.p.sigma*self.p.rep_ratio)
        
        y0 = np.zeros((5 *len(self.geo.popnodes)))
        y0[self.i.S]= self.geo.popnodes.copy()
        np.put(y0, ic_pts, y0[self.i.S].take(ic_pts) - I0)             # Tested OK
        np.put(y0, ic_pts + self.i.I[0], self.p.sigma*I0)                   # Tested OK
        np.put(y0, ic_pts + self.i.R[0], (1-self.p.sigma)*I0)               # Tested OK
        np.put(y0, ic_pts + self.i.B[0], y0[self.i.I].take(ic_pts)*self.p.theta/self.geo.popnodes.take(ic_pts)/self.p.muB)   # Tested OK
        np.put(y0, ic_pts + self.i.C[0], self.p.sigma*I0)                   # Tested OK
        
        return y0
        
    def build_ic(self):
        ic_pts = [150-1, 109-1, 119-1, 118-1]
        
        I0 = np.round(np.minimum(
            np.insert(1100*self.geo.popnodes.take(ic_pts[1:])/np.sum(self.geo.popnodes.take(ic_pts[1:])),0,1000)\
                    /(self.p.sigma*self.p.rep_ratio),
                self.geo.popnodes.take(ic_pts)))   # Tested OK
        
        y0 = np.zeros((5 *len(self.geo.popnodes)))
        y0[self.i.S]= self.geo.popnodes.copy()
        np.put(y0, ic_pts, y0[self.i.S].take(ic_pts) - I0)             # Tested OK
        np.put(y0, ic_pts + self.i.I[0], self.p.sigma*I0)                   # Tested OK
        np.put(y0, ic_pts + self.i.R[0], (1-self.p.sigma)*I0)               # Tested OK
        np.put(y0, ic_pts + self.i.B[0], y0[self.i.I].take(ic_pts)*self.p.theta/self.geo.popnodes.take(ic_pts)/self.p.muB)   # Tested OK
        np.put(y0, ic_pts + self.i.C[0], self.p.sigma*I0)                   # Tested OK
        
        return y0

    def plot(self):
        figsize = (20,20)

        r = Results(self.sib_r.y,self.i,self.s.geography, self.s)
        plt.tight_layout();
        fig, axes = plt.subplots(5, 2, figsize=figsize, squeeze = True);
        #axes = axes.flatten()
        #axes = axes[:-2]
        self.s.cases.plot(subplots=True, style='k', ax = axes, xlim=(self.s.t1i, self.s.t2f),linewidth=0.5);
        r.I_adm1_w.plot(subplots=True, ax=axes, xlim=(self.s.t1i, self.s.t2f), style='-' );
        
        fig, axes = plt.subplots(5, 2, figsize=figsize, squeeze = True);
        #axes = axes.flatten()
        #axes = axes[:-2]
        self.s.cases.cumsum().plot(subplots=True, style='k', ax = axes, xlim=(self.s.t1i, self.s.t2f),linewidth=0.5);
        r.C_adm1_w.plot(subplots=True, ax=axes, xlim=(self.s.t1i, self.s.t2f));
        
        plt.show()
        

class Results():
    def __init__(self, sib, setup):
        self.S = pd.DataFrame(sib[setup.i.S,:].T,columns=np.arange(0,setup.geo.nnodes), index=pd.date_range(setup.t1i, setup.t2f))
        self.I = pd.DataFrame(sib[setup.i.I,:].T,columns=np.arange(0,setup.geo.nnodes), index=pd.date_range(setup.t1i, setup.t2f))
        self.R = pd.DataFrame(sib[setup.i.R,:].T,columns=np.arange(0,setup.geo.nnodes), index=pd.date_range(setup.t1i, setup.t2f))
        self.B = pd.DataFrame(sib[setup.i.B,:].T,columns=np.arange(0,setup.geo.nnodes), index=pd.date_range(setup.t1i, setup.t2f))
        self.C = pd.DataFrame(sib[setup.i.C,:].T,columns=np.arange(0,setup.geo.nnodes), index=pd.date_range(setup.t1i, setup.t2f))
        
        
        self.S_adm1 = pd.DataFrame(0, columns=setup.geo.adm1_name, index=pd.date_range(setup.t1i, setup.t2f))
        self.I_adm1 = pd.DataFrame(0, columns=setup.geo.adm1_name, index=pd.date_range(setup.t1i, setup.t2f))
        self.R_adm1 = pd.DataFrame(0, columns=setup.geo.adm1_name, index=pd.date_range(setup.t1i, setup.t2f))
        self.B_adm1 = pd.DataFrame(0, columns=setup.geo.adm1_name, index=pd.date_range(setup.t1i, setup.t2f))
        self.C_adm1 = pd.DataFrame(0, columns=setup.geo.adm1_name, index=pd.date_range(setup.t1i, setup.t2f))
        for ind,dept in enumerate(setup.geo.adm1_name):
            for ws in np.nonzero(setup.geo.ws_adm1[:,ind])[0]:
                self.S_adm1.loc[:,dept] += setup.geo.ws_adm1[ws,ind] * self.S.loc[:][ws]
                self.I_adm1.loc[:,dept] += setup.geo.ws_adm1[ws,ind] * self.I.loc[:][ws]
                self.R_adm1.loc[:,dept] += setup.geo.ws_adm1[ws,ind] * self.R.loc[:][ws]
                self.B_adm1.loc[:,dept] += setup.geo.ws_adm1[ws,ind] * self.B.loc[:][ws]
                self.C_adm1.loc[:,dept] += setup.geo.ws_adm1[ws,ind] * self.C.loc[:][ws]
        self.C_adm1_w = self.C_adm1.resample('W-SAT').asfreq()
        self.I_adm1_w = self.C_adm1_w.diff()
        #self.I_adm1_w.iloc[0] =  self.C_adm1_w.iloc[0]
        

class ResultsDept():
    def __init__(self, sib, setup):
        self.S = pd.DataFrame(sib[setup.i.S,:].T,columns=setup.geo.adm1_name, index=pd.date_range(setup.t1i, setup.t2f))
        self.I = pd.DataFrame(sib[setup.i.I,:].T,columns=setup.geo.adm1_name, index=pd.date_range(setup.t1i, setup.t2f))
        self.R = pd.DataFrame(sib[setup.i.R,:].T,columns=setup.geo.adm1_name, index=pd.date_range(setup.t1i, setup.t2f))
        self.B = pd.DataFrame(sib[setup.i.B,:].T,columns=setup.geo.adm1_name, index=pd.date_range(setup.t1i, setup.t2f))
        self.C = pd.DataFrame(sib[setup.i.C,:].T,columns=setup.geo.adm1_name, index=pd.date_range(setup.t1i, setup.t2f))
        self.C_adm1_w = self.C.resample('W-SAT').asfreq()
        self.I_adm1_w = self.C_adm1_w.diff()
        #self.I_adm1_w.iloc[0] =  self.C_adm1_w.iloc[0]


class SIRB_Parameters():
    def __init__(self, theta, l, m, D, lam, rho, sigma, muB, beta0, aw):
        self.theta  = theta
        self.l      = l
        self.m      = m
        self.D      = D
        self.lam    = lam
        self.rho    = 1/(rho*365)
        self.sigma  = sigma
        self.muB    = muB
        self.beta0  = beta0
        self.aw     = aw
        
        # Fixed Parameters
        self.gamma     = 0.2                 # rate at which people recover from cholera (day^-1)
        self.rep_ratio = 1                   # reported to total (symptomatic) cases ratio --> asymptomatic cases do not get reported

        # Parameters fixed according to litterature and former research
        self.mu          = 1/(61.4*365)              # population natality and mortality rate (day^-1)
        self.alpha       = -np.log(0.98)*self.gamma  # mortality rate due to cholera (day^-1) (2% case fatality rate)
        self.epsilon_muB = 0                         # possible seasonality in mortality of bacteria (set to 0)
        
        # From code SERRA
        self.muB_wsd = np.full((365, ), 0.029268983866374) 
    @classmethod    
    def from_list(cls, param_list):
        """ Maintains compatility with enrico's prior"""
        return cls(    param_list[0],
                       param_list[1],
                       param_list[2],
                       param_list[3],
                       param_list[4],
                       param_list[5],
                       param_list[6],
                       param_list[7],
                       param_list[8],
                       param_list[9]
                  )

        
class Geography():
    def __init__(self, filename):
        self.adm1_name = ['Artibonite', 'Centre', 'Grande Anse', 'Nippes',
        'Nord', 'Nord-Est', 'Nord-Ouest', 'Ouest', 'Sud', 'Sud-Est']
        geodata = sio.loadmat(filename)
        self.nnodes = geodata['nnodes'].flatten()[0]
        self.x = geodata['X']
        self.y = geodata['Y']
        self.AD = geodata['AD']
        self.popnodes = geodata['POPnodes'].flatten()
        self.ws_adm1 = geodata['WS_dept']
        self.dist_road = geodata['dist_road']
        self.H = np.sum(self.popnodes)
        
    



class SIRB_ind():
    def __init__ (self, nnodes):
        self.S = np.arange(0*nnodes, 1*nnodes)
        self.I = np.arange(1*nnodes, 2*nnodes)
        self.R = np.arange(2*nnodes, 3*nnodes)
        self.B = np.arange(3*nnodes, 4*nnodes)
        self.C = np.arange(4*nnodes, 5*nnodes)
    




