#!/usr/bin/env python

import wuml
import numpy as np
import torch
import wplotlib
import torch.nn as nn
from torch.autograd import Variable
 
data = wuml.wData(xpath='../../data/data.comp3.csv', batch_size=20, 
					label_type='continuous', label_column_name='finalga_best', 
					row_id_with_label=0, columns_to_ignore=['id'])

[X_train, X_test, y_train, y_test] = wuml.split_training_test(data, data_name='Chem_decimated_imputed', 
										data_path='../data/', save_as='no saving',
										xdata_type="%.4f", ydata_type="%.4f", test_percentage=0.1)


X_train = wuml.center_and_scale(X_train)
X_test = wuml.center_and_scale(X_test)



weights = wuml.wData(xpath='../../data/comp3_weights.csv')
weights = weights.get_data_as('Tensor')

def costFunction(x, y, ŷ, ind):
	relu = nn.ReLU()

	#W = torch.squeeze(weights[ind])
	W = 1
	n = len(ind)
	ŷ = torch.squeeze(ŷ)
	y = torch.squeeze(y)

	penalty = torch.sum(relu(W*(ŷ - y)))/n	# This will penalize predictions higher than true labels
	loss = torch.sum(W*((y - ŷ)**2))/n + 0.3*penalty
	return loss


#bNet = wuml.basicNetwork(costFunction, data, networkStructure=[(1,'none')], max_epoch=6000, learning_rate=0.01)
bNet = wuml.basicNetwork(costFunction, X_train, networkStructure=[(30,'relu'),(1,'none')], max_epoch=6000, learning_rate=0.01)
#import pdb; pdb.set_trace()
bNet.train()

Ŷ = bNet(X_train, output_type='ndarray')
SR = wuml.summarize_regression_result(X_train.Y, Ŷ)
print(SR.avg_error())
print(SR.true_vs_predict(sort_based_on_label=True))
Tb = SR.true_vs_predict(sort_based_on_label=True)


Ŷtest = bNet(X_test, output_type='ndarray')
SR = wuml.summarize_regression_result(X_test.Y, Ŷtest)
print(SR.avg_error())
import pdb; pdb.set_trace()
