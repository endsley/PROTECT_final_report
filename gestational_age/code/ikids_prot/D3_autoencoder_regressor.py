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
import torch.nn as nn
from precision_recall import *

def costFunction(x, x̂, ẙ, y, ŷ, ind):	
#	x -> encoder -> x̂
#	x̂ -> encoder_linear_output -> ẙ	
#	x̂ -> decoder -> ŷ	
#	possible autoencoder objective λ could be 0
#	loss = (x - ŷ)ᒾ + λ * objective(ẙ, y)
#
#	This function can return 1 value or 3 values in a list
#	if return 1 value, just the loss
#	if return 3 values, [total_loss, reconstruction_loss, extra network from ẙ loss]
#
#	In this example, we perform both reconstruction and CE loss
#
	n = x.shape[0]
	d = x.shape[1]
	relu = nn.ReLU()
	R = 0.005*torch.sum((x - ŷ) ** 2)/(d*n)	#scaled by batch size times data dimension
	R2 = 30*torch.sum((y - ẙ) ** 2)/n	# 30 , 20, 10
	R3 = 0.6*torch.sum(relu((ŷ - 42)))/n	# if prediction above 43, its wrong
	R4 = 0.6*torch.sum(relu((23 - ŷ))/n)	# if prediction below 22, its wrong
	R5 = 1*(torch.sum(relu((ẙ - y)))*torch.sum(relu((37.5-y))))/(n)
	R6 = 1*(torch.sum(relu((y - 37)))*torch.sum(relu((37-ẙ))))/(n)
			# 1 doesn't seem to explode upward
	loss = R + R2 + R3 + R4 + R5 + R6
	return [loss, R, R2]



#	Data
data = wuml.wData(xpath='./data/D3_Imputed_Balanced_regression.csv', batch_size=32, 
					label_type='continuous', label_column_name='gestationAge',
					first_row_is_label=True)
[X_train, X_test, y_train, y_test] = wuml.split_training_test(data, test_percentage=0.2)
y_train_pre = X_train.pop_column('preterm_best')
y_test_pre = X_test.pop_column('preterm_best')

X_train.Data_preprocess()
X_test.Data_preprocess()

d = X_train.shape[1]


#	Objective 1
bottleneck_size = 28
AE = wuml.autoencoder(bottleneck_size, X_train, default_depth=3, costFunction=costFunction, # costFunction and costFunction2 both works
						EncoderStructure=[(1000,'relu'),(800,'relu'),(400,'relu'),(bottleneck_size,'none')], 	#bottleneck 28 out okay
						DecoderStructure=[(bottleneck_size,'relu'),(30,'relu'),(d,'relu')], 
						encoder_output_weight_structure=[(1,'none')],
						max_epoch=4000) 

AE.fit()

#	This is the objective network output Training error
ẙ = AE.objective_network(X_train)
gestational_precision_recall(ẙ, y_train_pre)
res = wuml.output_regression_result(y_train, ẙ, sort_by='error', print_out=['histograms', 'mean absolute error'])

#	This is the objective network output Test error
ẙ = AE.objective_network(X_test)
gestational_precision_recall(ẙ, y_test_pre)
tb1 = wuml.output_regression_result(y_test, ẙ, sort_by='error', print_out=['histograms', 'mean absolute error'] )
tb2 = wuml.output_regression_result(y_test, ẙ, sort_by='label', ascending=True, print_out=[] )
wuml.print_two_matrices_side_by_side(tb1, tb2, title1='Sort by worse error', title2='Sort by label')

