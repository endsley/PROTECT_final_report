#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')
import wuml
from proj_tools import *

#../data/D5_no_labels.csv

if len(sys.argv) < 2:
	raise ValueError("\n\tError : This program is designed to take a csv file with the pregnant women's profiles.")

#	load stuff
data = wuml.wData(xpath=sys.argv[1], first_row_is_label=True)
net = wuml.load_torch_network('./pk_files/D5_CE_500_1_10_explained.pk')

#	process data, run it through network and get its output, set it as label
data.Data_preprocess(mean=net.μ, std=net.σ)
[labels, prob_of_positive, gestages] = net(data, output_type='ndarray')
data.Y = gestages

#	explain
E = net.explainer(data)

