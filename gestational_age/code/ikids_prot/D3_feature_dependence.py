#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import wuml
from wuml.IO import *
import numpy as np
import torch
import wplotlib
import torch.nn as nn
from torch.autograd import Variable
 


data = wuml.wData(xpath='./data/D3_Imputed_Balanced_regression.csv', mv_columns_to_extra_data='preterm_best', 
						preprocess_data='center and scale', first_row_is_label=True)



jupyter_print('Features chosen')
jupyter_print(data.columns)

top_correlated_pair = wuml.feature_wise_correlation(data, n=30, get_top_corr_pairs=True)
jupyter_print('Most correlated feature pairs')
jupyter_print(top_correlated_pair, display_all_rows=True)


jupyter_print('Most correlated features to the label')
most_correlated_to_label = wuml.feature_wise_correlation(data, label_name='gestationAge', get_top_corr_pairs=True)
jupyter_print(most_correlated_to_label)


jupyter_print('Most Dependent HSIC feature pairs')
jupyter_print(wuml.feature_wise_HSIC(data, get_top_dependent_pairs=True))


jupyter_print('Most HSIC Dependent features to the label')
jupyter_print(wuml.feature_wise_HSIC(data, label_name='gestationAge', get_top_dependent_pairs=True))

