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
	n  = x.shape[0]
	λ = 0.01
	relu = nn.ReLU()
	R = λ*torch.sum((x - ŷ) ** 2)/(32*n)	#scaled by batch size times data dimension
	R2 = torch.sum((ẙ - y) ** 2)/(n)
	R3 = torch.sum(relu((ŷ - 43)))	# if prediction above 43, its wrong
	R4 = torch.sum(relu((22 - ŷ)))	# if prediction below 43, its wrong
	R5 = (torch.sum(relu((ẙ - y)))/n)*(torch.sum(relu(37 - y))/n)
	loss = R + R2 + R3 + R4 + R5
	return [loss, R, R2]




#	Data
data = wuml.wData(xpath='./data/data_1_imputed.csv', batch_size=32, preprocess_data='center and scale',
					label_type='continuous', label_column_name='gestationAge',
					first_row_is_label=True)
[X_train, X_test, y_train, y_test] = wuml.split_training_test(data, test_percentage=0.2)


#	Objective 1
AE = wuml.autoencoder(30, X_train, default_depth=3, costFunction=costFunction, # costFunction and costFunction2 both works
						max_epoch=4000, encoder_output_weight_structure=[(200,'relu'),(200,'relu'),(1,'none')] ) 

AE.fit()

#	This is the objective network output Training error
ẙ = AE.objective_network(X_train)
res = wuml.output_regression_result(y_train, ẙ, sort_by='error')

#	This is the objective network output Test error
ẙ = AE.objective_network(X_test)
res = wuml.output_regression_result(y_test, ẙ, sort_by='error')



import pdb; pdb.set_trace()
