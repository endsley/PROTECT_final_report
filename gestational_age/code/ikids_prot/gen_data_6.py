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



##	Here we split the data into train validate and test
keep_list = ['MEP' ,'BP3' ,'TCS' ,'PPB' ,'MIBP' ,'MPB' ,'MBP' ,'MECPP' ,'DCP25' ,'MHIBP','stress' ,'age_m' ,'BMI_prepreg' ,'income', 'preterm_best']

data = wuml.wData(xpath='./data/D3_Imputed_Balanced_regression.csv', only_keep_these_columns=keep_list, first_row_is_label=True, label_column_name='gestationAge', label_type='continuous')
[X_train, X_test, y_train, y_test] = wuml.split_training_test(data, test_percentage=0.1)
X_train.to_csv('./data/D6.csv', add_row_indices=False, include_column_names=True, float_format='%.4f')
X_test.to_csv('./data/D6_test.csv', add_row_indices=False, include_column_names=True, float_format='%.4f')





