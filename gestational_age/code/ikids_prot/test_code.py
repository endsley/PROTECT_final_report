#!/usr/bin/env python

import sys
import os

code_list = []
code_list.append('data_1_feature_dependency.py')
code_list.append('D3_feature_dependence.py')

#code_list.append('data_1_basic_regression.py')
#code_list.append('data_1_autoencoder_regressor.py')
#code_list.append('data_1_Neuralnet_regressor.py')
#code_list.append('data_1_Neuralnet_classifier.py')
#
#code_list.append('D2_NN_classifier.py')
#code_list.append('D2_NN_regressor.py')

#code_list.append('D3_NN_classifier.py')
#code_list.append('D3_NN_regressor.py')
#code_list.append('D3_NN_regressor2.py')

#code_list.append('D3_ComplexNet.py')
#code_list.append('D3_ComplexNet2.py')
#code_list.append('D3_ComplexNet3_80p.py')
#code_list.append('D3_ComplexNet3_60p.py')



print('Running data_stats folder')
for code in code_list:
	print('\tRunning ' + code)
	os.system('./' + code + ' disabled')

print('Make sure html folder exists')
if not os.path.exists('./html'): os.mkdir('./html')

print('Generate ipynb files')
for code in code_list:
	print('\tGenerating ' + code)
	os.system('p2j -o ' + code + ' > /dev/null 2>&1')
	ipynb = code.replace('.py', '.ipynb')
	os.system('jupyter nbconvert --to notebook --execute ' + ipynb + ' > /dev/null 2>&1')
	ran_ipynb = ipynb.replace('.ipynb', '.nbconvert.ipynb')
	os.system('mv ' + ran_ipynb + ' ' + ipynb)
	
	html_ipynb = ipynb.replace('.ipynb', '.html')
	os.system('mv ' + html_ipynb + ' ./html/' + html_ipynb)

