#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import wuml
from wuml.IO import *
from precision_recall import *
import numpy as np
import torch
import wplotlib
import torch.nn as nn
 

#	Basic Cross Entropy Loss on imbalanced data
def costFunction(x, y, ŷ, ind):
	CE_loss = nn.CrossEntropyLoss() #weird pytorch, dim of y is 1, and ŷ is 20x3
	return CE_loss(ŷ, y)



data = wuml.wData(xpath='./data/D1_imputed.csv', batch_size=32, columns_to_ignore=['gestationAge'], preprocess_data='center and scale',
					label_type='discrete', label_column_name='preterm_best', first_row_is_label=True)

[X_train, X_test, y_train, y_test] = wuml.split_training_test(data, test_percentage=0.1)

#	Objective 1 --------------------------------------------
bNet = wuml.basicNetwork(costFunction, X_train, networkStructure=[(600,'relu'),(600,'relu'),(600,'relu'),(2,'none')], max_epoch=4, learning_rate=0.01)
bNet.train(print_status=True)


#	This is the objective network output Training error
ẙ = bNet(X_train, output_type='ndarray', out_structural='1d_labels')	
res = wuml.summarize_classification_result(y_train, ẙ)


#	This is the objective network output Test error
ẙ = bNet(X_test, output_type='ndarray', out_structural='1d_labels')	
res = wuml.summarize_classification_result(y_test, ẙ)


