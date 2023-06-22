#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import wuml
from wuml.IO import *
from proj_tools import *
import numpy as np
import torch
import wplotlib
import torch.nn as nn
 



data = wuml.wData(xpath='./data/data_1_imputed.csv', batch_size=32, preprocess_data='center and scale',
					label_type='continuous', label_column_name='gestationAge',
					first_row_is_label=True)

wuml.jupyter_print('\n\nRun all regressors sorted by least test error')
result = wuml.run_every_regressor(data, alpha=0.1, gamma=0.05, l1_ratio=0.05)
wuml.jupyter_print(result['Train/Test Summary'])

model = result['RandomForest']

#	accuracy, precision, recall
ŷ = model(data)
gestational_precision_recall(ŷ, data.Y, print_out=True)
print('\n\n')

#	Show True label vs predicted label
YvY = model.show_true_v_predicted()
wuml.jupyter_print(YvY.df, display_all_rows=True)

#	feature importance
cnames = data.get_column_names_as_a_list()
model.plot_feature_importance('Feature Importance for Random Forest', cnames, xticker_rotate=90, ticker_fontsize=6)
model.output_sorted_feature_importance_table(data.columns)


