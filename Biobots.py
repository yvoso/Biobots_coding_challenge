# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 09:05:23 2016

@author: miffyvo
"""

import json
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

with open("/Users/miffyvo/Desktop/bioprint-data.json") as data_file:
    data = json.load(data_file)
    
    out = []
    for entry in data:
        u_email = entry['user_info']['email']
        u_serial = entry['user_info']['serial']
        pd_dead = entry['print_data']['deadPercent']
        pd_elasticity = entry['print_data']['elasticity']
        pd_live = entry['print_data']['livePercent']
        pi_cl_dur = entry['print_info']['crosslinking']['cl_duration']
        pi_cl_enabled = entry['print_info']['crosslinking']['cl_enabled']
        pi_cl_intensity = entry['print_info']['crosslinking']['cl_intensity']
        pi_input = entry['print_info']['files']['input']
        pi_output = entry['print_info']['files']['output']
        pi_extr1 = entry['print_info']['pressure']['extruder1']
        pi_extr2 = entry['print_info']['pressure']['extruder2']
        pi_height = entry['print_info']['resolution']['layerHeight']
        pi_num = entry['print_info']['resolution']['layerNum']
        pi_wellplate = entry['print_info']['wellplate']
        out.append((u_email, u_serial, pd_dead, pd_elasticity, pd_live,
                    pi_cl_dur, pi_cl_enabled, pi_cl_intensity,
                    pi_input, pi_output, pi_extr1, pi_extr2, pi_height,
                    pi_num, pi_wellplate))
    cols = ('email', 'serial', 'dead', 'elasticity', 'live',
            'duration', 'enabled', 'intensity', 'input', 'output',
            'extruder1', 'extruder2', 'layer_height', 'layer_num', 'wellplate')
    pddata = pd.DataFrame.from_records(out, columns = cols)    
    
    #print (pddata)
    
    regr = linear_model.LinearRegression()
    
    excluded_cols = [
        'email', 
        'dead', 
        'live', 
        'elasticity', 
        'input', 
        'output'
    ]
    xvars = [
        'duration', 
        'enabled', 
        'intensity', 
        'extruder1', 
        'extruder2', 
        'layer_height', 
        'layer_num', 
        'wellplate'
    ]
    
    length = pddata.shape[0]
    
    # inspect which x has the strongest relationship with y
    y = pddata['live'].reshape(length, 1)
    for j in xvars:
        x = pddata[j].reshape(length, 1)
        regr.fit(x, y)
        plt.scatter(x, y,  color='black')
        plt.plot(x, regr.predict(x), color='blue', linewidth=3)
        plt.show()
        plt.savefig('X_Y_relationship.png')

    # inspect if any pair of x are linearly correlated
    nxvars = len(xvars)
    for i in range(0, 8):
        for j in range(i, 8):
            if i != j:
                y = pddata[xvars[i]].reshape(length, 1)
                x = pddata[xvars[j]].reshape(length, 1)
                regr.fit(x, y)
                plt.scatter(x, y,  color='black')
                plt.plot(x, regr.predict(x), color='blue', linewidth=3)
                plt.xticks(())
                plt.yticks(())
                plt.xlabel(xvars[i], fontsize=18)
                plt.ylabel(xvars[j], fontsize=16)
                plt.show()
                plt.savefig('X_X_relationship.png')
                
    #Overall, the relationships between different variables seem to be lacking obvious correlation,
    #However some data type that are continous would require different kind of analysis other than regression.
            
    
    
