#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')
import wuml
from proj_tools import *

wuml.set_terminal_print_options(precision=3)
all_files = wuml.list_all_files_in_directory('./pk_files', with_extension='.pk')
all_files.sort()

summary_table = np.empty((0, 10))
for f in all_files:
	net = wuml.load_torch_network('./pk_files/' + f, load_as_cpu_or_gpu='cpu')
	if 'D5' in f: 
		data_names = 'D5'
	elif 'D4' in f: 
		data_names = 'D4_subset_of_D3'
	elif 'D3' in f: 
		data_names = 'D3_Imputed_Balanced_regression'
#
	data = wuml.wData(xpath='../data/' + data_names + '.csv', batch_size=32, 
						label_type='continuous', label_column_name='gestationAge',
						mv_columns_to_extra_data='preterm_best', preprocess_data='center and scale',
						first_row_is_label=True)
#
	[labels, prob_of_positive, gestages] = net(data, output_type='ndarray')
	result_summary = [f] + display_results(data.Y, gestages, data.xDat[0], labels, prob_of_positive, display=False)		# ['Accᶜ', 'Precisionᶜ', 'Recallᶜ', 'AUCᶜ', 'Accᴿ', 'Precisionᴿ', 'Recallᴿ', 'Avg L1 error', 'Avg Lᨖ error']
	summary_table = np.vstack((summary_table, result_summary))

cnames = ['file', 'Accᶜ', 'Precisionᶜ', 'Recallᶜ', 'AUCᶜ', 'Accᴿ', 'Precisionᴿ', 'Recallᴿ', 'Avg L1 error', 'Avg Lᨖ error']
summary_table = wuml.ensure_DataFrame(summary_table, columns=cnames)
wuml.jupyter_print(summary_table, display_all_rows=True, display_all_columns=True)

