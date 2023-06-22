#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import numpy as np
import wuml
from wuml.type_check import *
import pandas as pd
from sklearn.metrics import accuracy_score


def gestational_precision_recall(ŷ, y, print_out=False):
	ŷ = wuml.ensure_numpy(ŷ)
	y = wuml.ensure_numpy(y)

	uy = np.unique(ŷ)
	if len(uy) != 2 or (0 not in uy) or (1 not in uy):		# then we have a regression problem
		ŷ = (ŷ < 37).astype(int)	# convert into binvary values with 1 as 


	uy2 = np.unique(y)
	if len(uy2) != 2 or (0 not in uy2) or (1 not in uy2):		# then we have a regression problem
		y = (y < 37).astype(int)	# convert into binvary values with 1 as 

	Acc= accuracy_score(y, ŷ)
	P = wuml.precision(y, ŷ)
	R = wuml.recall(y, ŷ)

	if print_out: wuml.jupyter_print('\nPrecision: %.3f, Recall : %.3f'%(P,R))

	Δy = np.absolute(ŷ - y)
	cnames = np.array(['yᴿ', 'ŷᴿ', 'Δyᴿ'])
	Yjoin = np.hstack((y, ŷ, Δy))
	df = pd.DataFrame(Yjoin, columns=cnames)

	return [Acc, P, R, df]


def display_results(y, ŷ, yᶜ, ŷᶜ, prob_positive, display=True):
	PP = wuml.ensure_wData(ensure_column(np.round(prob_positive[:,1], 4)), column_names="P(X=1)")
	auc = wuml.binary_auc(np.squeeze(yᶜ), prob_positive[:,1])

	CR = wuml.summarize_classification_result(yᶜ, ŷᶜ, print_out=None)
	tb1 = CR.true_vs_predict(add_to_label='ᶜ')
	outStr = '\n\t---------------------\n\n'
	outStr += '\t Classifier Accuracy : %.4f\n'%CR.accuracy
	outStr += '\t Classifier Precision: %.4f\n'%CR.Precision
	outStr += '\t Classifier Recall: %.4f\n'%CR.Recall
	outStr += '\t Classifier AUC: %.4f\n'%auc
	

	[Acc, P, R, tb3] = gestational_precision_recall(ŷ, yᶜ, print_out=False)
	outStr += '\t Regressor accuracy: %.4f\n'%Acc
	outStr += '\t Regressor Precision: %.4f\n'%P
	outStr += '\t Regressor Recall: %.4f\n'%R
	
	
	regRes = wuml.summarize_regression_result(y, ŷ, print_out=None, rounding=2)
	TB = regRes.true_vs_predict(sort_by='error')
	worse_case_error = TB.X[0,2]

	main_table = regRes.true_vs_predict()
	main_table.append_columns(tb1)
	main_table.append_columns(PP)
	main_table.append_columns(tb3)
	main_table.sort_by('Δy')
	
	outStr += '\t Mean absolute error: %.4f\n'%regRes.mean_absolute_error
	if display:
		regRes.error_histogram()
		wuml.jupyter_print(outStr)
		wuml.jupyter_print(main_table, display_all_rows=True, display_all_columns=True)

	outlist = [CR.accuracy, CR.Precision, CR.Recall, auc, Acc, P, R, regRes.mean_absolute_error, worse_case_error]
	outlist = list(np.around(np.array(outlist),3))
	return outlist

