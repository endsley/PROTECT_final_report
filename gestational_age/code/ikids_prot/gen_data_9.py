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


#-------------Regression ----------------------------------------------------
##	Here we remove some features, and performed basic imputation+rebalance for Regression
#keep_list = ['MEP' ,'BP3' ,'TCS' ,'PPB' ,'MIBP' ,'MPB' ,'MBP' ,'MECPP' ,'DCP25' ,'MHIBP','stress' ,'age_m' ,'BMI_prepreg' ,'income',  'gestationAge']
#data = wuml.wData(xpath='../../data/IKIDS_PROTECT.csv', only_keep_these_columns=keep_list, first_row_is_label=True, label_column_name='preterm_best', label_type='discrete')

rm_list = ['id' , 'cohort', 'edu_p' ,'marital_m' ,'race' ,'insur_m' ,'hypertension_ever' ,'hypertension_currpreg' ,'preeclampsia' ,'gestdiab_prevpreg' ,'gestdiab_currpreg' ,'diabetes_ever' ,'asthma_ever' ,'asthma_preg', 'MECPTP', 'MEHHTP', 'MONP']
data = wuml.wData(xpath='../../data/IKIDS_PROTECT.csv', batch_size=32, 
					label_type='discrete', label_column_name='preterm_best', 
					first_row_is_label=True, columns_to_ignore=rm_list)


data_imputed = wuml.impute(data)	
data_imputed.to_csv('./data/D9.csv', add_row_indices=False, include_column_names=True, float_format='%.4f')


