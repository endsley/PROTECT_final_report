#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')
import wuml
from proj_tools import *


#	Which data
data_name = 'D5'
#data_name = 'D4_subset_of_D3'
#data_name = 'D3_Imputed_Balanced_regression'

#	Which network
#net = wuml.load_torch_network('./pk_files/D3_CE_2000_99.pk', load_as_cpu_or_gpu='cpu')
#net = wuml.load_torch_network('./pk_files/D4_CE_2000_99.pk', load_as_cpu_or_gpu='cpu')
#net = wuml.load_torch_network('./pk_files/D5_CE_2000_1_10.pk', load_as_cpu_or_gpu='cpu')
net = wuml.load_torch_network('./pk_files/D5_CE_500_1_10.pk')


#------------------------------------------

#	data setup
data = wuml.wData(xpath='../data/' + data_name + '.csv', batch_size=32, 
					label_type='continuous', label_column_name='gestationAge',
					mv_columns_to_extra_data='preterm_best',
					first_row_is_label=True)

[X_train, X_test, y_train, y_test] = wuml.split_training_test(data, test_percentage=0.2)
X_train.Data_preprocess()
X_test.Data_preprocess(mean=X_train.μ, std=X_train.σ)


#	Training
[labels, prob_of_positive, gestages] = net(X_train, output_type='ndarray')
display_results(X_train.Y, gestages, X_train.xDat[0], labels, prob_of_positive)

#	Test
[labels, prob_of_positive, gestages] = net(X_test, output_type='ndarray')
display_results(X_test.Y, gestages, X_test.xDat[0], labels, prob_of_positive)

E = wuml.explainer(data, net, explainer_algorithm='shap', which_model_output_to_use=2)
exp = E(X_test[0:200,:])

wuml.save_torch_network(net, './pk_files/D5_CE_500_1_10_explained.pk')


