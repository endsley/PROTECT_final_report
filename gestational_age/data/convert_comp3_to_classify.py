#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')
import wuml 

##	Convert
#data = wuml.wData(xpath='./data.comp3.csv', batch_size=20, 
#					label_type='continuous', label_column_name='finalga_best', 
#					row_id_with_label=0, columns_to_ignore=['id'])
#
#class_labels = (data.Y < 37).astype(int)
#data.replace_label(class_labels)
#data.to_csv('./data.comp3_classify.csv', include_column_names=True, float_format='%.4f')



#	read
data = wuml.wData(xpath='./data.comp3_classify.csv', batch_size=20, 
					label_type='discrete', label_column_name='finalga_best', 
					row_id_with_label=0, columns_to_ignore=['id'])

import pdb; pdb.set_trace()
