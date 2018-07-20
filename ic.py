import spotpy 
import datetime
import numpy as np
import matplotlib.pyplot as plt
from setup import SIRB_Setup, Results, ResultsDept
from sirb import SIRB
from misc import *
import pandas as pd

#setup = SIRB_Setup().model_ws()
#setup.t1i = datetime.date(2010, 10, 24)
#setup.t2f = datetime.date(2016, 8, 15) 
#setup.t1f = setup.t2f 
#setup.p.beta_h2h = 0.00001

#models = ['norm', 'eisen', 'eisen-h2h', 'norm-h2h']
#for model_str in models:
    #model = SIRB(setup, model_str)
    #result  = Results(model.run().y, setup)
    #S = result.S.loc[setup.t2f].as_matrix()
    #I = result.I.loc[setup.t2f].as_matrix()
    #R = result.R.loc[setup.t2f].as_matrix()
    #B = result.B.loc[setup.t2f].as_matrix()
    #C = result.C.loc[setup.t2f].as_matrix()
    #IC = np.concatenate((S, I, R, B, C))
    #np.save('IC_'+model_str+'-ws', IC)
    
    
    
    
setup = SIRB_Setup().model_dept()
setup.t1i = datetime.date(2010, 10, 24)
setup.t2f = datetime.date(2016, 8, 13) 
setup.t1f = setup.t2f 
setup.p.beta_h2h = 0.00001

models = ['norm', 'eisen', 'eisen-h2h', 'norm-h2h']
for model_str in models:
    model = SIRB(setup, model_str)
    result  = ResultsDept(model.run().y, setup)
    S = result.S.loc[setup.t2f].as_matrix()
    I = result.I.loc[setup.t2f].as_matrix()
    R = result.R.loc[setup.t2f].as_matrix()
    B = result.B.loc[setup.t2f].as_matrix()
    C = result.C.loc[setup.t2f].as_matrix()
    IC = np.concatenate((S, I, R, B, C))
    np.save('IC_'+model_str+'-dept', IC)
