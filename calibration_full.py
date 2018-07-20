import spotpy
import sys
import datetime, time
import spotpy
import sys
import numpy as np
from setup import SIRB_Setup, Results
from sirb import SIRB
from misc import *
import pandas as pd
import uuid
import random
import scipy.stats as sst

model_str = sys.argv[1]
scale = sys.argv[2]
n = int(sys.argv[3])



print(""" 
    *********************************************************
             CALIBRATION (Model = {0} & n = {1})
    *********************************************************
    """.format(model_str, n))


class spotpy_setup(object):
    def __init__(self, model_str):
        if (scale == 'ws'):
            self.setup = SIRB_Setup().model_ws()
        elif(scale == 'dept'):
            self.setup = SIRB_Setup().model_dept()
            
 
        
        self.setup.t1i = datetime.date(2010, 10, 24)
        self.setup.t1f = datetime.date(2016, 8, 13) 
        self.setup.t2f = datetime.date(2016, 12, 30)
        
        # time for comparison
        self.ti = self.setup.t1f
        self.tf = self.setup.t2f
        
        self.setup.p.theta = 1
                
        self.model = SIRB(self.setup, model_str)
        
        self.params = [#spotpy.parameter.Uniform('theta',0, 3, optguess = random.uniform(0, 3)),
                        #spotpy.parameter.Uniform('l',0, 0.1, optguess = random.uniform(0, 0.1)),
                        spotpy.parameter.Uniform('m',0, 0.3, optguess = random.uniform(0, 0.3)),
                        spotpy.parameter.Uniform('D',0, 600, optguess = random.uniform(0, 600)),
                        spotpy.parameter.Uniform('rho',1, 7, optguess = random.uniform(1, 7)),
                        spotpy.parameter.Uniform('sigma',0.01, 0.3, optguess = random.uniform(0.01, 0.3)),
                        spotpy.parameter.Uniform('muB',1/365, 1, optguess = random.uniform(1/365, 1))]
        
        # Parameter definition
        if (model_str == 'eisen'):
            self.params.append(spotpy.parameter.Uniform('beta_u',0, 10, optguess = random.uniform(0, 10)))
        elif (model_str == 'norm'):
            self.params.extend((spotpy.parameter.Uniform('beta_u',0, 10, optguess = random.uniform(0, 10)),
                           spotpy.parameter.Uniform('lambda',0, 4, optguess = random.uniform(0, 4))))
        elif (model_str == 'eisen-h2h'):
            self.params.extend((spotpy.parameter.Uniform('beta_u'  ,0, 10, optguess = random.uniform(0, 10)),
                           spotpy.parameter.Uniform('beta_h2h',0, 0.5, optguess = random.uniform(0, 0.5))))
        elif (model_str == 'norm-h2h'):
            self.params.extend((spotpy.parameter.Uniform('beta_u',0, 10, optguess = random.uniform(0, 10)),
                           spotpy.parameter.Uniform('lambda',0, 8, optguess = random.uniform(0, 8)),
                           spotpy.parameter.Uniform('beta_h2h',0, 0.5, optguess = random.uniform(0, 0.5))))
                           
                           


        
 
    def parameters(self):
        return spotpy.parameter.generate(self.params)
    
    def simulation(self, vector):
        print('Parameter is', vector, )
        #self.model.p.theta = vector[0]
        #self.model.p.l = vector[1]
        self.model.p.m = vector[0]
        self.model.p.D = vector[1]
        self.model.p.rho = vector[2]
        self.model.p.sigma = vector[3]
        self.model.muB = vector [4]

        if (model_str == 'eisen'):
            self.model.p.beta0 = vector[5]
        elif (model_str == 'norm'):
            self.model.p.beta0 = vector[5]
            self.model.p.lam = vector[6]
        elif (model_str == 'norm-h2h'):
            self.model.p.beta0 = vector[5]
            self.model.p.lam = vector[6]
            self.model.p.beta_h2h = vector[7]
        elif (model_str == 'eisen-h2h'):
            self.model.p.beta0 = vector[5]
            self.model.p.beta_h2h = vector[6]
            
        try:
            result = self.model.run().y
        except:
            result = None
        
        try:
            if (scale == 'ws'):
                C =      pd.DataFrame(result[self.setup.i.C,:].T, 
                                    columns=np.arange(0,self.setup.geo.nnodes), 
                                    index=pd.date_range(self.setup.t1i, self.setup.t2f)) 
                C_adm1 = pd.DataFrame(0, 
                                    columns=self.setup.geo.adm1_name,
                                    index=pd.date_range(self.setup.t1i, self.setup.t2f))
            
                for ind,dept in enumerate(self.setup.geo.adm1_name):
                    for ws in np.nonzero(self.setup.geo.ws_adm1[:,ind])[0]:
                        C_adm1.loc[:,dept] += self.setup.geo.ws_adm1[ws,ind] * C.loc[:][ws]
            
                C_adm1_w = C_adm1.resample('W-SAT').asfreq()
                I_adm1_w = C_adm1_w.diff()
                I_adm1_w.iloc[0] = 0 #instead of NaN
            #elif (scale == 'dept'):
                #C = pd.DataFrame(result[self.setup.i.C,:].T,
                                 #columns=self.setup.geo.adm1_name, 
                                 #index=pd.date_range(self.setup.t1i, self.setup.t2f))
                #C_adm1_w = C.resample('W-SAT').asfreq()
                #I_adm1_w = C_adm1_w.diff()
                #I_adm1_w.iloc[0] = 0
            elif (scale == 'dept'):
                #I = pd.DataFrame(result[self.setup.i.I,:].T,columns=self.setup.geo.adm1_name, index=pd.date_range(self.setup.t1i, self.setup.t2f))
                #I_adm1_w =  I.resample('W-SAT').sum() # THIS WAS WRONG: I IS NOT THE NUMBER OF NEW CASES BUT THE NUMBER OF INFECTED !!!
                
                C = pd.DataFrame(result[self.setup.i.C,:].T,columns=self.setup.geo.adm1_name, index=pd.date_range(self.setup.t1i, self.setup.t2f))
                C_adm1_w = C.resample('W-SAT').asfreq()
                I_adm1_w = C_adm1_w.diff()

            print(bcolors.OKGREEN + '>>> SIMULATION SUCCESS' + bcolors.ENDC)
        except:
             print(bcolors.FAIL + '>>> SIMULATION FAILED' + bcolors.ENDC)
             I_adm1_w = pd.DataFrame(-100000000, 
                                     columns=self.setup.geo.adm1_name, 
                                     index=pd.date_range(self.setup.t1i, self.setup.t2f)).resample('W-SAT').asfreq()
        
        simulation = I_adm1_w[self.ti:self.tf].iloc[1:] #first value as no sense regarding the diff
        simulation = simulation.as_matrix().flatten()
        where_are_NaNs = np.isnan(simulation)
        simulation[where_are_NaNs] = 0
        return simulation
    
    def evaluation(self):
        evaluation = self.setup.cases[self.ti:self.tf].iloc[1:].as_matrix().flatten()
        where_are_NaNs = np.isnan(evaluation)
        evaluation[where_are_NaNs] = 0
        return evaluation
    
    def objectivefunction(self,simulation, evaluation):
        #obj = -spotpy.objectivefunctions.rmse(evaluation,simulation)
        #obj = -np.sqrt(((simulation - evaluation) ** 2).mean())
        #obj = spotpy.likelihoods.gaussianLikelihoodMeasErrorOut(evaluation,simulation)/10  # Variance  ?? TODO
        sigma = 100         #Error of meszurement
        #obj = -np.sum(sst.norm.logpdf(evaluation, loc=simulation, scale=sd))
        error =  simulation - evaluation
        obj =  (len(error)/2)*np.log(2*np.pi) - len(error)*np.log(sigma) - 1/2*sigma**(-2)*np.sum(error**2)
        print(bcolors.OKBLUE, "--- --- Objective is ", obj, bcolors.ENDC)
        return obj
    
spotpy_setup=spotpy_setup(model_str)


dbname = 'DREAM_' + model_str + '_' + scale + '_' + str(n) + '_' + time.strftime("%Y-%m-%d::%H:%M:%S") +  '_' +str(uuid.uuid4())[:2]


sampler = spotpy.algorithms.dream(spotpy_setup, dbname=dbname, dbformat='csv', alt_objfun=None, save_sim = False)
nChains                = 4
convergence_limit      = 0
runs_after_convergence = 0

r_hat = sampler.sample(n,nChains=nChains,convergence_limit=convergence_limit, 
        runs_after_convergence=runs_after_convergence)


