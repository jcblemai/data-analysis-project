import scipy.io as sio
import numpy as np
import pandas as pd
import datetime, time
import numpy.linalg as lia
np.set_printoptions(threshold=np.nan, suppress=True)
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.integrate


class SIRB():
    
    def __init__(self, setup, model_str):
        self.s = setup
        self.y0 = setup.y0
        self.p = setup.p
        self.i = setup.i
        
        # Convert dataframe to np.array for speeed, maybe not ness
        self.rainfall = self.s.rainfall[self.s.t1i:self.s.t2f+datetime.timedelta(days=1)].as_matrix()
        self.forcing_cases = np.zeros_like(self.rainfall)
        for ind, t in enumerate(pd.date_range(self.s.t1i, self.s.t1f+datetime.timedelta(days=1))):
            self.forcing_cases[ind][:] = self.s.cases.loc[t:].iloc[0]/7
        
        self.t = np.arange(0,  self.s.t2f.toordinal() + 1 - self.s.t1i.toordinal())
        
        self.t_force = max(self.s.t1f.toordinal() - self.s.t1i.toordinal(),0)
        
        self.dy = np.zeros((5 *len(self.s.geo.popnodes)))
        
        self.beta_force = np.empty((self.t_force, self.s.geo.nnodes))
        
        if (model_str == 'eisen'):
            self.model = self.SIB_eisenberger
        elif (model_str == 'norm'):
            self.model = self.SIB_norm
        elif (model_str == 'enrico'):
            self.model = self.SIB_enrico
        elif (model_str == 'legacy'):
            self.model = self.SIB_legacy
        elif (model_str == 'eisen-h2h'):
            self.model = self.SIB_eisenberger_h2h
        elif (model_str == 'norm-h2h'):
            self.model = self.SIB_norm_h2h
        
        
        
        #print(">>> SIB model Ready")
        
    def run(self):
        tic = time.time()
        sib = scipy.integrate.solve_ivp(fun = self.model,
                                        t_span = (self.t[0], self.t[-1]+1),
                                        y0 = self.y0,
                                        method='RK45',
                                        t_eval = self.t,
                                        max_step = 1.0)
                                        
        print(">>> Simulation done in ", time.time()-tic)
        return sib
    
    
        
    
    def SIB_enrico(self, t, y): 
        """
        The 'normal' SIB model:
            - With awareness
            - K = 1
            - Rainfall acts on bacterial concentration
            Should normally be exactly like enrico's one but it is not the case, 
            so we need to investigate with the right notebook #TODO
        """
        t = int(t)
        temp  = np.dot(self.s.fluxes, y[self.i.B] / (y[self.i.B]+1))
        temp3 = np.dot(self.s.HPtH,   y[self.i.B]) * self.p.l
        
        if (t < self.t_force):
            S = self.s.geo.popnodes - y[self.i.I] - y[self.i.R]
            FI = self.forcing_cases[t]/self.p.sigma/S
            self.beta_force[t,:] = FI
        
        else:
            FI = ((1-self.p.m)*y[self.i.B]/(1 + y[self.i.B]) + temp*self.p.m) * (self.p.beta0 * np.exp(-(y[self.i.C]/self.s.geo.popnodes)/self.p.aw)) 
        TT = FI * y[self.i.S] 
    
        
        self.dy[self.i.S] = self.p.mu * (self.s.geo.popnodes - y[self.i.S]) - TT + self.p.rho*y[self.i.R]
        self.dy[self.i.I] = self.p.sigma * TT - (self.p.gamma + self.p.alpha + self.p.mu) * y[self.i.I]
        self.dy[self.i.R] = (1 - self.p.sigma) * TT + self.p.gamma * y[self.i.I] - (self.p.mu + self.p.rho) * y[self.i.R]
        self.dy[self.i.B] =  -(self.p.muB_wsd + self.p.l) * y[self.i.B] + (self.p.theta/self.s.geo.popnodes) \
        * (1+self.p.lam * self.rainfall[t]) * y[self.i.I] + temp3
                             
        self.dy[self.i.C] = self.p.sigma * TT
        return self.dy
    

    
    def SIB_legacy(self, t, y):
        rainfall =  self.rainfall[int(t)]
        temp  = np.dot(self.s.fluxes, y[self.i.B] / (y[self.i.B]+1))
        temp3 = np.dot(np.dot(self.s.HPtH, y[self.i.B]), self.p.l)
        TT = ((1-self.p.m)*y[self.i.B]/(1 + y[self.i.B]) + temp*self.p.m) * y[self.i.S] * (self.p.beta0 * np.exp(-(y[self.i.C]/self.s.geo.popnodes)/self.p.aw))
        dy = np.zeros((5 *len(self.s.geo.popnodes)))
        dy[self.i.S] = self.p.mu * (self.s.geo.popnodes - y[self.i.S]) - TT + self.p.rho*y[self.i.R]
        dy[self.i.I] = self.p.sigma * TT - (self.p.gamma + self.p.alpha + self.p.mu) * y[self.i.I]
        dy[self.i.R] = (1 - self.p.sigma) * TT + self.p.gamma * y[self.i.I] - (self.p.mu + self.p.rho) * y[self.i.R]
        dy[self.i.B] =  -(self.p.muB_wsd + self.p.l) * y[self.i.B] + (self.p.theta/self.s.geo.popnodes) * (1+self.p.lam * rainfall) * y[self.i.I] + temp3
                             
        dy[self.i.C] = self.p.sigma * TT
        return dy
    
    def SIB_norm(self, t, y): 
        """
        The 'normal' SIB model:
            - No awareness
            - K = 1
            - Rainfall acts on bacterial concentration
        """
        t = int(t)
        temp  = np.dot(self.s.fluxes, y[self.i.B] / (y[self.i.B]+1))
        temp3 = np.dot(self.s.HPtH,   y[self.i.B]) * self.p.l
        
        if (t < self.t_force):
            S = self.s.geo.popnodes - y[self.i.I] - y[self.i.R]
            FI = self.forcing_cases[t]/self.p.sigma/S
            self.beta_force[t,:] = FI
        
        else:
            FI = ((1-self.p.m)*y[self.i.B]/(1 + y[self.i.B]) + temp*self.p.m) * self.p.beta0 
        TT = FI * y[self.i.S] 
    
        
        self.dy[self.i.S] = self.p.mu * (self.s.geo.popnodes - y[self.i.S]) - TT + self.p.rho*y[self.i.R]
        self.dy[self.i.I] = self.p.sigma * TT - (self.p.gamma + self.p.alpha + self.p.mu) * y[self.i.I]
        self.dy[self.i.R] = (1 - self.p.sigma) * TT + self.p.gamma * y[self.i.I] - (self.p.mu + self.p.rho) * y[self.i.R]
        self.dy[self.i.B] =  -(self.p.muB_wsd + self.p.l) * y[self.i.B] + (self.p.theta/self.s.geo.popnodes) \
        * (1+self.p.lam * self.rainfall[t]) * y[self.i.I] + temp3
        self.dy[self.i.C] = self.p.sigma * TT
        
        return self.dy
    
    def SIB_norm_h2h(self, t, y): 
        """
        The 'normal' SIB model:
            - No awareness
            - K = 1
            - Rainfall acts on bacterial concentration
        """
        t = int(t)
        temp  = np.dot(self.s.fluxes, y[self.i.B] / (y[self.i.B]+1))
        temp3 = np.dot(self.s.HPtH,   y[self.i.B]) * self.p.l
        
        temp4 = np.dot(self.s.fluxes, y[self.i.I])
        
        if (t < self.t_force):
            S = self.s.geo.popnodes - y[self.i.I] - y[self.i.R]
            FI = self.forcing_cases[t]/self.p.sigma/S
            self.beta_force[t,:] = FI
        
        else:
            FI = ((1-self.p.m)*y[self.i.B]/(1 + y[self.i.B]) + temp *self.p.m) * self.p.beta0 + \
                 ((1-self.p.m)*y[self.i.I]                   + temp4*self.p.m) * self.p.beta_h2h 
        TT = FI * y[self.i.S] 
    
        
        self.dy[self.i.S] = self.p.mu * (self.s.geo.popnodes - y[self.i.S]) - TT + self.p.rho*y[self.i.R]
        self.dy[self.i.I] = self.p.sigma * TT - (self.p.gamma + self.p.alpha + self.p.mu) * y[self.i.I]
        self.dy[self.i.R] = (1 - self.p.sigma) * TT + self.p.gamma * y[self.i.I] - (self.p.mu + self.p.rho) * y[self.i.R]
        self.dy[self.i.B] =  -(self.p.muB_wsd + self.p.l) * y[self.i.B] + (self.p.theta/self.s.geo.popnodes) \
        * (1+self.p.lam * self.rainfall[t]) * y[self.i.I] + temp3
        self.dy[self.i.C] = self.p.sigma * TT
        
        return self.dy
    
    def SIB_eisenberger_h2h(self, t, y): 
        """
        The 'eisenberger' SIB model, a textbook adaption of the eisenberger model
        """
        t = int(t)
        temp  = np.dot(self.s.fluxes, y[self.i.B] / (y[self.i.B]+1))
        temp2 = np.dot(self.s.fluxes, self.rainfall[t])
        temp3 = np.dot(self.s.HPtH,   y[self.i.B]) * self.p.l
        temp4 = np.dot(self.s.fluxes, y[self.i.I])
        
        if (t < self.t_force):
            S = self.s.geo.popnodes - y[self.i.I] - y[self.i.R]
            FI = self.forcing_cases[t]/self.p.sigma/S
            self.beta_force[t,:] = FI
        
        else:
            FI = ((1-self.p.m)*self.rainfall[t]*y[self.i.B]/(1 + y[self.i.B]) + temp2* temp*self.p.m) * self.p.beta0 + \
                 ((1-self.p.m)*y[self.i.I]                   + temp4*self.p.m) * self.p.beta_h2h

        TT = FI * y[self.i.S] 
    
        
        self.dy[self.i.S] = self.p.mu * (self.s.geo.popnodes - y[self.i.S]) - TT + self.p.rho*y[self.i.R]
        self.dy[self.i.I] = self.p.sigma * TT - (self.p.gamma + self.p.alpha + self.p.mu) * y[self.i.I]
        self.dy[self.i.R] = (1 - self.p.sigma) * TT + self.p.gamma * y[self.i.I] - (self.p.mu + self.p.rho) * y[self.i.R]
        self.dy[self.i.B] =  -(self.p.muB_wsd + self.p.l) * y[self.i.B] + (self.p.theta/self.s.geo.popnodes) * y[self.i.I] + temp3
        self.dy[self.i.C] = self.p.sigma * TT
        
        return self.dy
    
        
        
    #@jit
    def SIB_eisenberger(self, t, y): 
        """
        The 'eisenberger' SIB model, a textbook adaption of the eisenberger model
        """
        t = int(t)
        temp  = np.dot(self.s.fluxes, y[self.i.B] / (y[self.i.B]+1))
        temp2 = np.dot(self.s.fluxes, self.rainfall[t])
        temp3 = np.dot(self.s.HPtH,   y[self.i.B]) * self.p.l
        
        
        if (t < self.t_force):
            S = self.s.geo.popnodes - y[self.i.I] - y[self.i.R]
            FI = self.forcing_cases[t]/self.p.sigma/S
            self.beta_force[t,:] = FI
        
        else:
            FI = ((1-self.p.m)*self.rainfall[t]*y[self.i.B]/(1 + y[self.i.B]) + temp2* temp*self.p.m) * self.p.beta0 
        TT = FI * y[self.i.S] 
    
        
        self.dy[self.i.S] = self.p.mu * (self.s.geo.popnodes - y[self.i.S]) - TT + self.p.rho*y[self.i.R]
        self.dy[self.i.I] = self.p.sigma * TT - (self.p.gamma + self.p.alpha + self.p.mu) * y[self.i.I]
        self.dy[self.i.R] = (1 - self.p.sigma) * TT + self.p.gamma * y[self.i.I] - (self.p.mu + self.p.rho) * y[self.i.R]
        self.dy[self.i.B] =  -(self.p.muB_wsd + self.p.l) * y[self.i.B] + (self.p.theta/self.s.geo.popnodes) * y[self.i.I] + temp3
        self.dy[self.i.C] = self.p.sigma * TT
        
        return self.dy
    


