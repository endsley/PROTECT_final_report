#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')
import wuml
from proj_tools import *




#	data setup
data = wuml.wData(xpath='./data/D3_Imputed_Balanced_regression.csv', batch_size=32, 
					label_type='continuous', label_column_name='gestationAge',
					mv_columns_to_extra_data='preterm_best',
					first_row_is_label=True)

[X_train, X_test, y_train, y_test] = wuml.split_training_test(data, test_percentage=0.2)
X_train.Data_preprocess()
X_test.Data_preprocess()


##	load network
##net = wuml.load_torch_network('./gestNet_400_60.pk')
net = wuml.load_torch_network('./gestNet_2000_60.pk')
#net = wuml.load_torch_network('./gestNet_1000_60.pk')
##net = wuml.load_torch_network('./gestNet.pk')
#net = wuml.load_torch_network('./gestNet_800_60p.pk', load_as_cpu_or_gpu='cpu')

#	Training
[labels, prob_of_positive, gestages] = net(X_train, output_type='ndarray')
display_results(X_train.Y, gestages, X_train.xDat[0], labels, prob_of_positive)

#	Test
[labels, prob_of_positive, gestages] = net(X_test, output_type='ndarray')
display_results(X_test.Y, gestages, X_test.xDat[0], labels, prob_of_positive)

E = wuml.explainer(data, net, explainer_algorithm='shap', which_model_output_to_use=2)
exp = E(X_train)
exp = E(X_test)
