#!/usr/bin/env python

import wuml 
import numpy as np
import scipy.stats
from wplotlib import histograms
from wplotlib import lines


data = wuml.wData('../../data/data.comp3.csv', label_column_name='finalga_best', 
					label_type='continuous', row_id_with_label=0)

H1 = histograms(data.Y, num_bins=20, title='Gestational Age Histogram', facecolor='blue', α=0.5, path=None)
H1 = histograms(data.Y, num_bins=20, title='Gestational Age Histogram', facecolor='blue', α=0.5, path=None, ylogScale=True )

sample_weights = wuml.get_likelihood_weight(data.Y)
H = histograms(sample_weights.X, num_bins=20, title='Sample Weight Histogram', facecolor='blue', α=0.5, path=None, ylogScale=True )

print(wuml.output_two_columns_side_by_side(data.Y, sample_weights.X, labels=['Y','W'], rounding=3))
sample_weights.to_csv('../../data/comp3_weights.csv', include_column_names=False)
