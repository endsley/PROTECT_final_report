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
import wplotlib




##-------------Classification ----------------------------------------------------
###	Here we remove some features, and performed basic imputation+rebalance for classification
#rm_list = ['id' , 'gestationAge', 'cohort', 'edu_p' ,'marital_m' ,'race' ,'insur_m' ,'hypertension_ever' ,'hypertension_currpreg' ,'preeclampsia' ,'gestdiab_prevpreg' ,'gestdiab_currpreg' ,'diabetes_ever' ,'asthma_ever' ,'asthma_preg', 'MECPTP', 'MEHHTP', 'MONP']
#
#data = wuml.wData(xpath='../../data/IKIDS_PROTECT.csv', batch_size=32, 
#					label_type='discrete', label_column_name='preterm_best', 
#					first_row_is_label=True, columns_to_ignore=rm_list)
#
#data_imputed = wuml.impute(data)	
#
#rebalancer = wuml.rebalance_data(data_imputed, method='smote')
#data_balanced = rebalancer.balanced_data
#data_balanced.to_csv('./data/D3_Imputed_Balanced_classify.csv', add_row_indices=False, include_column_names=True, float_format='%.4f')



#-------------Regression ----------------------------------------------------
##	Here we remove some features, and performed basic imputation+rebalance for Regression
rm_list = ['id' , 'cohort', 'edu_p' ,'marital_m' ,'race' ,'insur_m' ,'hypertension_ever' ,'hypertension_currpreg' ,'preeclampsia' ,'gestdiab_prevpreg' ,'gestdiab_currpreg' ,'diabetes_ever' ,'asthma_ever' ,'asthma_preg', 'MECPTP', 'MEHHTP', 'MONP']

data = wuml.wData(xpath='../../data/IKIDS_PROTECT.csv', batch_size=32, 
					label_type='discrete', label_column_name='preterm_best', 
					first_row_is_label=True, columns_to_ignore=rm_list)


data_imputed = wuml.impute(data)	

rebalancer = wuml.rebalance_data(data_imputed, method='oversampling')
data_balanced = rebalancer.balanced_data

data_balanced.swap_label('gestationAge')
data_balanced.to_csv('./data/D3_Imputed_Balanced_regression.csv', add_row_indices=False, include_column_names=True, float_format='%.4f')


