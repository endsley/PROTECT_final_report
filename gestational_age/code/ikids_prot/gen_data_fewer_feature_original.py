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
#rm_list = ['id' , 'cohort', 'edu_p' ,'marital_m' ,'race', 'DCP24' ,'insur_m' ,'hypertension_ever' ,'hypertension_currpreg' ,'preeclampsia' ,'gestdiab_prevpreg' ,'gestdiab_currpreg' ,'diabetes_ever' ,'asthma_ever' ,'asthma_preg', 'MECPTP', 'MEHHTP', 'MONP', 'preterm_best']
#
#data = wuml.wData(xpath='../../data/IKIDS_PROTECT.csv', batch_size=32, 
#					first_row_is_label=True, columns_to_ignore=rm_list)


keep_list = ['MEP' ,'BP3' ,'TCS' ,'PPB' ,'MIBP' ,'MPB' ,'MBP' ,'MECPP' ,'DCP25' ,'MHIBP','stress' ,'age_m' ,'BMI_prepreg' ,'income', 'marital_m', 'gestationAge', 'preterm_best']
data = wuml.wData(xpath='../../data/IKIDS_PROTECT.csv', only_keep_these_columns=keep_list, first_row_is_label=True)

data_imputed = wuml.impute(data)	
data_imputed.to_csv('./data/fewer_feature_original.csv', add_row_indices=False, include_column_names=True, float_format='%.4f')


