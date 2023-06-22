#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import wuml
import numpy as np



#data = wuml.wData(xpath='./data/D3_Imputed_Balanced_regression.csv', batch_size=32, 
#					label_type='discrete', label_column_name='preterm_best',
#					columns_to_ignore='gestationAge', first_row_is_label=True)
#
#D1_list = []
#D2_list = []
#for i in range(20):
#	[X_train, X_test, y_train, y_test] = wuml.split_training_test(data, test_percentage=0.2)
#	
#	Train_negative = X_train.get_all_samples_from_a_class(0)
#	Train_positive = X_train.get_all_samples_from_a_class(1)
#	Test_positive = X_test.get_all_samples_from_a_class(1)
#	
#	D1 = wuml.mmd(Train_positive, Test_positive)
#	D2 = wuml.mmd(Train_negative, Test_positive)
#	D1_list.append(D1)
#	D2_list.append(D2)
#
#print('MMD Distances for D3')
#print('\tMean Distance between train positive and test positive = %.4f ± %.3f, max = %.3f, min = %.3f'%(np.mean(D1_list), np.std(D1_list), np.max(D1_list), np.min(D1_list)))
#print('\tMean Distance between train negative and test positive = %.4f ± %.3f, max = %.3f, min = %.3f'%(np.mean(D2_list), np.std(D2_list), np.max(D2_list), np.min(D2_list)))


# --D9 ------------------------------------------------------------------------------


data = wuml.wData(xpath='./data/D9.csv', batch_size=32, 
					label_type='discrete', label_column_name='preterm_best',
					columns_to_ignore='gestationAge', first_row_is_label=True)

D1_list = []
D2_list = []
for i in range(20):
	[X_train, X_test, y_train, y_test] = wuml.split_training_test(data, test_percentage=0.2)
	
	Train_negative = X_train.get_all_samples_from_a_class(0)
	Train_positive = X_train.get_all_samples_from_a_class(1)
	Test_positive = X_test.get_all_samples_from_a_class(1)

	D1 = wuml.mmd(Train_positive, Test_positive)
	D2 = wuml.mmd(Train_negative, Test_positive)
	D1_list.append(D1)
	D2_list.append(D2)

print('MMD Distances for D3')
print('\tMean Distance between train positive and test positive = %.4f ± %.3f, max = %.3f, min = %.3f'%(np.mean(D1_list), np.std(D1_list), np.max(D1_list), np.min(D1_list)))
print('\tMean Distance between train negative and test positive = %.4f ± %.3f, max = %.3f, min = %.3f'%(np.mean(D2_list), np.std(D2_list), np.max(D2_list), np.min(D2_list)))

# --------------------------------------------------------------------------------

D1_list = []
D2_list = []
Train = wuml.wData(xpath='./data/D7.csv', batch_size=32, 
					label_type='discrete', label_column_name='preterm_best',
					columns_to_ignore='gestationAge', first_row_is_label=True)

Test = wuml.wData(xpath='./data/D7_test.csv', batch_size=32, 
					label_type='discrete', label_column_name='preterm_best',
					columns_to_ignore='gestationAge', first_row_is_label=True)



Train_negative = Train.get_all_samples_from_a_class(0)
Train_positive = Train.get_all_samples_from_a_class(1)
Test_positive = Test.get_all_samples_from_a_class(1)

D1 = wuml.mmd(Train_positive, Test_positive)
D2 = wuml.mmd(Train_negative, Test_positive)

print('MMD Distances for D7')
print('\tDistance between train positive and test positive = %.4f'%D1)
print('\tDistance between train negative and test positive = %.4f'%D2)


# --------------------------------------------------------------------------------

Train = wuml.wData(xpath='./data/D8.csv', batch_size=32, 
					label_type='discrete', label_column_name='preterm_best',
					columns_to_ignore='gestationAge', first_row_is_label=True)

Test = wuml.wData(xpath='./data/D8_test.csv', batch_size=32, 
					label_type='discrete', label_column_name='preterm_best',
					columns_to_ignore='gestationAge', first_row_is_label=True)



Train_negative = Train.get_all_samples_from_a_class(0)
Train_positive = Train.get_all_samples_from_a_class(1)
Test_positive = Test.get_all_samples_from_a_class(1)

D1 = wuml.mmd(Train_positive, Test_positive)
D2 = wuml.mmd(Train_negative, Test_positive)

print('MMD Distances for D8')
print('\tDistance between train positive and test positive = %.4f'%D1)
print('\tDistance between train negative and test positive = %.4f'%D2)

