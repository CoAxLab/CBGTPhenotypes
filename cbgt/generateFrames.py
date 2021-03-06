

# massive helper function to create a list of datatables (one per sim) and a paramtable of parameter vectors
# The function "generateFrames" takes in two parameters, the "results" and "decisions" variables created in
# the code cell above it.  It outputs two variables:
#
#    - "paramtable" which is a dataframe of the parameter vectors for each sim (automatically limited as to only
#       contain the parameters that are actually different for different sims).
#    - "datatables" which is a list of data frames, one dataframe per sim with one row per trial.
#       It contains columns that indicate the times (absolute time within the sim run) of stimuli/decisions/rewards,
#       as well as the duration (decision-stimulus) with and without the reward delay. It contains the network's
#       decision, as well as the correct decision and the reward.
#       Those last two will require adjustments depending on what sort of sim we are running.
#       It also contains both the mean and integral firing rates for all populations for all 3 time intervals
#       (stimulus-decision, decision-reward, stimulus-reward).

import os, sys
import pandas as pd
import numpy as np
import random
import math

import cbgt.netgen as ng

def generateFrames(results,decisions):
    
    datatables = []
    
    paramtable = pd.DataFrame()
    index = 0
    for confignum in range(0,len(results)):
        for repnum in range(0,len(results[confignum])):
            paramvector = pd.DataFrame([[index,confignum,repnum]],columns=['index','confignum','repnum'])
            index += 1
            for key,value in results[confignum][repnum].items():
                if isinstance(value, (float,int,str)):
                    paramvector[key] = value
            paramvector.set_index('index',inplace=True)
            paramtable = paramtable.append(paramvector)
            
            datatable = pd.DataFrame()
            
            result = results[confignum][repnum]
            
            stagecount = len(decisions[confignum][repnum])
            
            for i in range(0, stagecount):
                
                row = pd.DataFrame([[i]],columns=['trial'])
                
                
                if decisions[confignum][repnum][i]['pathvals'] != None and len(decisions[confignum][repnum][i]['pathvals']) > 0:
                    row['decision'] = decisions[confignum][repnum][i]['pathvals'][0]
                else:
                    row['decision'] = None
                
                row['stimulusstarttime'] = decisions[confignum][repnum][i]['time'] - decisions[confignum][repnum][i]['delay']
                row['decisiontime'] = decisions[confignum][repnum][i]['time']
                row['decisionduration'] = decisions[confignum][repnum][i]['delay']
                row['decisiondurationplusdelay'] = decisions[confignum][repnum][i]['delay'] + 300
                row['rewardtime'] = decisions[confignum][repnum][i]['time'] + 300
                
                # NOTE: has to match code in netgen
                row['correctdecision'] = int(i>=20)
                row['reward'] = row.apply(lambda x: int(x.decision == x.correctdecision), axis=1)
                
                msd = result['popfreqs'][result['popfreqs']['Time (ms)'].between(row.loc[0,'stimulusstarttime'], row.loc[0,'decisiontime'])].drop(columns='Time (ms)').mean().to_frame().T
                isd = msd.copy().multiply(row.loc[0,'decisiontime'] - row.loc[0,'stimulusstarttime']).divide(1000)
                msd.columns = ["msd_" + str(col) for col in msd.columns]
                row = row.join(msd)
                isd.columns = ["isd_" + str(col) for col in isd.columns]
                row = row.join(isd)
                
                
                mdr = result['popfreqs'][result['popfreqs']['Time (ms)'].between(row.loc[0,'decisiontime'], row.loc[0,'rewardtime'])].drop(columns='Time (ms)').mean().to_frame().T
                idr = mdr.copy().multiply(row.loc[0,'rewardtime'] - row.loc[0,'decisiontime']).divide(1000)
                mdr.columns = ["mdr_" + str(col) for col in mdr.columns]
                row = row.join(mdr)
                idr.columns = ["idr_" + str(col) for col in idr.columns]
                row = row.join(idr)
                
                
                msr = result['popfreqs'][result['popfreqs']['Time (ms)'].between(row.loc[0,'stimulusstarttime'], row.loc[0,'rewardtime'])].drop(columns='Time (ms)').mean().to_frame().T
                isr = msr.copy().multiply(row.loc[0,'rewardtime'] - row.loc[0,'stimulusstarttime']).divide(1000)
                msr.columns = ["msr_" + str(col) for col in msr.columns]
                row = row.join(msr)
                isr.columns = ["isr_" + str(col) for col in isr.columns]
                row = row.join(isr)
                
                
                datatable = datatable.append(row)
            
            
            datatable.set_index('trial',inplace=True)
            datatables.append(datatable)
    

    nunique = paramtable.apply(pd.Series.nunique)
    cols_to_keep = pd.Index(['confignum','repnum']).append(nunique[nunique > 1].index).unique()
    paramtable = paramtable[cols_to_keep]
    
    return (paramtable,datatables)
