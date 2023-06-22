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

dat = ['fewer_feature_original', 'D10', 'D13']

# -----------------------------

for D in dat:
	wuml.jupyter_print('Data %s -------------------'%D)
	data = wuml.wData(xpath='./data/' + D + '.csv', first_row_is_label=True)
	wuml.jupyter_print(data.shape)
	wuml.jupyter_print(data.columns)
	
	data.get_feature_histograms('gestationAge', num_bins=10, ylogScale=False)
	gAges = np.round(data.get_columns('gestationAge', return_type='ndarray'))
	premat = np.round(data.get_columns('preterm_best', return_type='ndarray'))
	
	wuml.get_label_stats(gAges, print_stat=True)
	wuml.get_label_stats(premat, print_stat=True)





