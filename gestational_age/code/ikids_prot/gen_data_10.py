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
rm_list = ['id' , 'cohort', 'edu_p' ,'marital_m' ,'race', 'DCP24' ,'insur_m' ,'hypertension_ever' ,'hypertension_currpreg' ,'preeclampsia' ,'gestdiab_prevpreg' ,'gestdiab_currpreg' ,'diabetes_ever' ,'asthma_ever' ,'asthma_preg', 'MECPTP', 'MEHHTP', 'MONP', 'preterm_best']

data = wuml.wData(xpath='../../data/IKIDS_PROTECT.csv', batch_size=32, 
					first_row_is_label=True, columns_to_ignore=rm_list)

data_imputed = wuml.impute(data)	
positive_cases = data_imputed.get_all_samples_based_on_condition_of_column('gestationAge', condition='less than', conditional_value=37)
negative_cases = data_imputed.get_all_samples_based_on_condition_of_column('gestationAge', condition='greater than or equal to', conditional_value=37)

positive_cases.to_csv('./data/D10_positive_cases.csv', add_row_indices=False, include_column_names=True, float_format='%.4f')
negative_cases.to_csv('./data/D10_negative_cases.csv', add_row_indices=False, include_column_names=True, float_format='%.4f')


