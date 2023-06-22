#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')
import wuml 
from wuml import jupyter_print


#	read
data = wuml.wData(xpath='../../data/data.comp3_classify.csv', batch_size=20, 
					label_type='discrete', label_column_name='finalga_best', 
					row_id_with_label=0, columns_to_ignore=['id'])

#data = wuml.center_and_scale(data)
jupyter_print('\n\nRun all classifier sorted by Accuracy')
results = wuml.run_every_classifier(data, y=data.Y, order_by='Test')
jupyter_print(results['Train/Test Summary'])

import pdb; pdb.set_trace()
