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
 

def costFunction(x, y, ŷ, ind):
	n = x.shape[0]
	relu = nn.ReLU()
	R = torch.sum((y - ŷ) ** 2)
	R3 = 10*torch.sum(relu((ŷ - 42)))	# if prediction above 43, its wrong
	R4 = 10*torch.sum(relu((23 - ŷ)))	# if prediction below 22, its wrong
	R5 = 50*(torch.sum(relu((ŷ - y)))*torch.sum(relu((37-y))))/(n*n)
	loss = R + R3 + R4 + R5
	return loss


def costFunction2(x, y, ŷ, ind):
	n = x.shape[0]
	relu = nn.ReLU()
	R = torch.sum((y - ŷ) ** 2)
	R3 = 10*torch.sum(relu((ŷ - 42)))	# if prediction above 43, its wrong
	R4 = 10*torch.sum(relu((23 - ŷ)))	# if prediction below 22, its wrong
	R5 = 60*(torch.sum(relu((ŷ - y)))*torch.sum(relu((37-y))))/(n*n)
	loss = R + R3 + R4 + R5
	return loss

def costFunction3(x, y, ŷ, ind):
	n = x.shape[0]
	relu = nn.ReLU()
	R = torch.sum((y - ŷ) ** 2)
	R3 = 10*torch.sum(relu((ŷ - 42)))	# if prediction above 43, its wrong
	R4 = 10*torch.sum(relu((23 - ŷ)))	# if prediction below 22, its wrong
	R5 = 80*(torch.sum(relu((ŷ - y)))*torch.sum(relu((37-y))))/(n*n)
	loss = R + R3 + R4 + R5
	return loss


def costFunction4(x, y, ŷ, ind):
	n = x.shape[0]
	relu = nn.ReLU()
	R = torch.sum((y - ŷ) ** 2)
	R3 = 10*torch.sum(relu((ŷ - 42)))	# if prediction above 43, its wrong
	R4 = 10*torch.sum(relu((23 - ŷ)))	# if prediction below 22, its wrong
	R5 = 130*(torch.sum(relu((ŷ - y)))*torch.sum(relu((37-y))))/(n*n)
	loss = R + R3 + R4 + R5
	return loss


def costFunction5(x, y, ŷ, ind):
	n = x.shape[0]
	relu = nn.ReLU()
	R = torch.sum((y - ŷ) ** 2)
	R3 = 10*torch.sum(relu((ŷ - 42)))	# if prediction above 43, its wrong
	R4 = 10*torch.sum(relu((23 - ŷ)))	# if prediction below 22, its wrong
	R5 = 160*(torch.sum(relu((ŷ - y)))*torch.sum(relu((37-y))))/(n*n)
	loss = R + R3 + R4 + R5
	return loss

def costFunction6(x, y, ŷ, ind):
	n = x.shape[0]
	relu = nn.ReLU()
	R = torch.sum((y - ŷ) ** 2)
	R3 = 20*torch.sum(relu((ŷ - 42)))	# if prediction above 43, its wrong
	R4 = 20*torch.sum(relu((23 - ŷ)))	# if prediction below 22, its wrong
	R5 = 200*(torch.sum(relu((ŷ - y)))*torch.sum(relu((37-y))))/(n*n)
	loss = R + R3 + R4 + R5
	return loss



data = wuml.wData(xpath='./data/D2_Imputed_Balanced_regression.csv', batch_size=32, 
					label_type='continuous', label_column_name='gestationAge', first_row_is_label=True)

[X_train, X_test, y_train, y_test] = wuml.split_training_test(data, test_percentage=0.1)
y_train_pre = X_train.pop_column('preterm_best')
y_test_pre = X_test.pop_column('preterm_best')

X_train.Data_preprocess()
X_test.Data_preprocess()


#	Objective 1 --------------------------------------------
#fun_list = [costFunction2, costFunction3, costFunction4, costFunction5, costFunction6]
fun_list = [costFunction3]

for foo in fun_list:
	bNet = wuml.basicNetwork(foo, X_train, networkStructure=[(600,'relu'),(600,'relu'),(600,'relu'),(1,'none')], max_epoch=4000, learning_rate=0.01)
	bNet.train(print_status=True)
	
	#	This is the objective network output Training error
	ẙ = bNet(X_train, output_type='ndarray')	
	print(foo.__name__)
	gestational_precision_recall(ẙ, y_train_pre)
	res = wuml.output_regression_result(y_train, ẙ, print_out=['mean absolute error'])
	
	#	This is the objective network output Test error
	ẙ = bNet(X_test, output_type='ndarray')	
	gestational_precision_recall(ẙ, y_test_pre)
	res = wuml.output_regression_result(y_test, ẙ, print_out=['mean absolute error', 'true v predict labels'])


