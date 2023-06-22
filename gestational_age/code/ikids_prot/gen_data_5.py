#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')


import wuml
import numpy as np
import torch
import wplotlib



##	Here we remove some features, and performed basic imputation+rebalance for Regression
keep_list = ['MEP' ,'BP3' ,'TCS' ,'PPB' ,'MIBP' ,'MPB' ,'MBP' ,'MECPP' ,'DCP25' ,'MHIBP','stress' ,'age_m' ,'BMI_prepreg' ,'income', 'preterm_best', 'gestationAge']

data = wuml.wData(xpath='./data/D3_Imputed_Balanced_regression.csv', only_keep_these_columns=keep_list, first_row_is_label=True)
data.to_csv('./data/D5.csv', add_row_indices=False, include_column_names=True, float_format='%.4f')


#A = [281.823963 , 114.463519 , 67.457493 , 48.309062 , 27.894448 , 18.550369 , 12.048283 , 11.394775 , 11.046140 , 8.591341 , 7.738448 , 7.395923 , 6.806019 , 6.397900 , 6.167886 , 5.497413 , 5.442741 , 5.388066 , 4.947283 , 4.533360 , 4.498739 , 4.005764 , 3.843377 , 3.821230 , 3.776849]
#np.cumsum(A)/np.sum(A)





