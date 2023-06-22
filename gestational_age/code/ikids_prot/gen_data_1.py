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

#	Here we remove some features, and performed basic imputation for regression
#rm_list = ['id' ,'cohort' ,'preterm_best' ,'edu_p' ,'marital_m' ,'race' ,'insur_m' ,'hypertension_ever' ,'hypertension_currpreg' ,'preeclampsia' ,'gestdiab_prevpreg' ,'gestdiab_currpreg' ,'diabetes_ever' ,'asthma_ever' ,'asthma_preg', 'MECPTP', 'MEHHTP', 'MONP']
#
#data = wuml.wData(xpath='../../data/IKIDS_PROTECT.csv', batch_size=32, 
#					label_type='continuous', label_column_name='gestationAge', 
#					first_row_is_label=True, columns_to_ignore=rm_list)
#wuml.missing_data_stats(data, save_plots=False)

#data.to_csv('./data/data_1_missing.csv', add_row_indices=False, include_column_names=True, float_format='%.4f')
#data_imputed = wuml.impute(data)	
#data_imputed.to_csv('./data/data_1_imputed.csv', add_row_indices=False, include_column_names=True, float_format='%.4f')


#-----------------------------------------------------------------
##	Here we remove some features, and performed basic imputation for regression (Data includes the preterm labels)
rm_list = ['id' ,'cohort', 'edu_p' ,'marital_m' ,'race' ,'insur_m' ,'hypertension_ever' ,'hypertension_currpreg' ,'preeclampsia' ,'gestdiab_prevpreg' ,'gestdiab_currpreg' ,'diabetes_ever' ,'asthma_ever' ,'asthma_preg', 'MECPTP', 'MEHHTP', 'MONP']

data = wuml.wData(xpath='../../data/IKIDS_PROTECT.csv', batch_size=32, 
					label_type='continuous', label_column_name='gestationAge', 
					first_row_is_label=True, columns_to_ignore=rm_list)


#Y2 = data.get_columns('preterm_best')	# need to remove and then added back in after imputation
#data.delete_column('preterm_best')
data_imputed = wuml.impute(data)	
#data_imputed.append_columns(Y2)
import pdb; pdb.set_trace()

data_imputed.to_csv('./data/D1_imputed.csv', add_row_indices=False, include_column_names=True, float_format='%.4f')




#-----------------------------------------------------------------



##	Here we remove some features, and performed basic imputation for Classification
#rm_list = ['id' ,'cohort' , 'gestationAge', 'edu_p' ,'marital_m' ,'race' ,'insur_m' ,'hypertension_ever' ,'hypertension_currpreg' ,'preeclampsia' ,'gestdiab_prevpreg' ,'gestdiab_currpreg' ,'diabetes_ever' ,'asthma_ever' ,'asthma_preg', 'MECPTP', 'MEHHTP', 'MONP']
# 
#
#data = wuml.wData(xpath='../../data/IKIDS_PROTECT.csv', batch_size=32, 
#					label_type='discrete', label_column_name='preterm_best', 
#					first_row_is_label=True, columns_to_ignore=rm_list)
#
#D = wuml.get_label_stats(data)
#
#import pdb; pdb.set_trace()
##data.to_csv('./data/data_1_missing.csv', add_row_indices=False, include_column_names=True, float_format='%.4f')
#data_imputed = wuml.impute(data)	
#data_imputed.to_csv('./data/data_1_classify_imputed.csv', add_row_indices=False, include_column_names=True, float_format='%.4f')
#
#
#import pdb; pdb.set_trace()

