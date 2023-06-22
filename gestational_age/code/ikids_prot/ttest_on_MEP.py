#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

from proj_tools import *
import wuml
import numpy as np
import torch
import torch.nn as nn
from scipy import stats

data = wuml.wData(xpath='./data/D3_Imputed_Balanced_regression.csv', 
					label_type='discrete', label_column_name='preterm_best', first_row_is_label=True)

X0 = data.get_all_samples_from_a_class(0)
X1 = data.get_all_samples_from_a_class(1)

#key_factors_D3 = ['MEP' ,'BP3' ,'TCS' ,'PPB' ,'MIBP' ,'MPB' ,'MECPP' ,'MBP' ,'stress', 'BMI_prepreg', 'DCP25', 'TCB', 'age_m', 'MHIBP', 'MEOHP', 'MCOP', 'MEHHP', 'MBZP']
key_factors_D5 = ['BP3' ,'MEP' ,'MBP' ,'MIBP' ,'MPB' ,'MHIBP' ,'TCS' ,'DCP25' ,'BMI_prepreg' ,'PPB' ,'MECPP' ,'stress' ,'age_m' ,'income']

for factor in key_factors_D5:
	print('\nT test for %s'%factor)
	MEP0 = np.squeeze(X0.get_columns([factor]).X)
	MEP1 = np.squeeze(X1.get_columns([factor]).X)
	Tstats = stats.ttest_ind(MEP0, MEP1)
	
	wuml.jupyter_print('\tNormal Birth mean %s = %.4f ± %.3f\n'%	(factor, np.mean(MEP0), np.std(MEP0) ))
	wuml.jupyter_print('\tNormal Birth median %s = %.4f'%	(factor, np.mean(MEP0) ))
	wuml.jupyter_print('\tAbnormal Birth mean %s = %.4f ± %.3f'%	(factor, np.mean(MEP1), np.std(MEP1) ))
	wuml.jupyter_print('\tAbnormal Birth median %s = %.4f'%	(factor, np.mean(MEP1) ))
	if Tstats[1] < 0.05:
		wuml.jupyter_print('\tp value: %.4f <------------- significant'%Tstats[1])
	else:
		wuml.jupyter_print('\tp value: %.4f'%Tstats[1])



