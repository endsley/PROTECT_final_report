#!/usr/bin/env python
import os
import sys
if os.path.exists('/home/chieh/code/wPlotLib'):
	sys.path.insert(0,'/home/chieh/code/wPlotLib')
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')

import wuml 
from wplotlib import scatter

wuml.set_terminal_print_options(precision=3)

#	Use flow to model the fewer feature original case but normalize data
data = wuml.wData(xpath='../ikids_prot/data/fewer_feature_original.csv', batch_size=32, 
					label_column_name='gestationAge', label_type='continuous', 
					preprocess_data='center and scale', 
					first_row_is_label=True)
data.swap_label('preterm_best')


#----------------------	between 22.5 to 26
S1 = data.get_all_samples_based_on_condition_of_column('gestationAge', condition='less than', conditional_value=26)
k = wuml.KDE(S1)
D1 = k.generate_samples(460)

newData = D1.get_all_samples_based_on_condition_of_column('gestationAge', condition='in between', upper_bound=26, lower_bound=22.5)
#newData.get_column_stats('gestationAge', float_round2Int=True)


#----------------------	between 26 to 30
S1 = data.get_all_samples_based_on_condition_of_column('gestationAge', condition='in between', upper_bound=31, lower_bound=26)
#S1.get_column_stats('gestationAge', float_round2Int=True)

k = wuml.KDE(S1)
D1 = k.generate_samples(720)

D_25_30 = D1.get_all_samples_based_on_condition_of_column('gestationAge', condition='in between', upper_bound=31, lower_bound=26)
#D_25_30.get_column_stats('gestationAge', float_round2Int=True)
newData.append_rows(D_25_30)



#----------------------	between 30 to 32
S1 = data.get_all_samples_based_on_condition_of_column('gestationAge', condition='in between', upper_bound=33, lower_bound=29)
#S1.get_column_stats('gestationAge', float_round2Int=True)

k = wuml.KDE(S1)
D1 = k.generate_samples(320)

D_30_32 = D1.get_all_samples_based_on_condition_of_column('gestationAge', condition='in between', upper_bound=33, lower_bound=29)
#D_30_32.get_column_stats('gestationAge', float_round2Int=True)
newData.append_rows(D_30_32)



#----------------------	between 32 to 36
S1 = data.get_all_samples_based_on_condition_of_column('gestationAge', condition='in between', upper_bound=36, lower_bound=32)
#S1.get_column_stats('gestationAge', float_round2Int=True)

k = wuml.KDE(S1)
D1 = k.generate_samples(320)

D_31_36 = D1.get_all_samples_based_on_condition_of_column('gestationAge', condition='in between', upper_bound=36, lower_bound=32)
#D_31_36.get_column_stats('gestationAge', float_round2Int=True)
newData.append_rows(D_31_36)

#----------------------	between 36 to 37
S1 = data.get_all_samples_based_on_condition_of_column('gestationAge', condition='in between', upper_bound=37.5, lower_bound=35)

k = wuml.KDE(S1)
D1 = k.generate_samples(300)

D_35_37 = D1.get_all_samples_based_on_condition_of_column('gestationAge', condition='in between', upper_bound=37.5, lower_bound=35)
newData.append_rows(D_35_37)



#----------------------	between 37 to 39
S1 = data.get_all_samples_based_on_condition_of_column('gestationAge', condition='in between', upper_bound=39, lower_bound=37)

k = wuml.KDE(S1)
D1 = k.generate_samples(500)

D_35_37 = D1.get_all_samples_based_on_condition_of_column('gestationAge', condition='in between', upper_bound=39, lower_bound=37)
newData.append_rows(D_35_37)
#newData.get_column_stats('gestationAge', float_round2Int=True)


#----------------------	between 39 to 41
S1 = data.get_all_samples_based_on_condition_of_column('gestationAge', condition='in between', upper_bound=41, lower_bound=39)


k = wuml.KDE(S1)
D1 = k.generate_samples(500)

D_35_37 = D1.get_all_samples_based_on_condition_of_column('gestationAge', condition='in between', upper_bound=41, lower_bound=39)
newData.append_rows(D_35_37)


#----------------------	between 41 to 43
S1 = data.get_all_samples_based_on_condition_of_column('gestationAge', condition='in between', upper_bound=43, lower_bound=41)


k = wuml.KDE(S1)
D1 = k.generate_samples(500)

D_35_37 = D1.get_all_samples_based_on_condition_of_column('gestationAge', condition='in between', upper_bound=43, lower_bound=41)
newData.append_rows(D_35_37)


#----------------------	43
S1 = data.get_all_samples_based_on_condition_of_column('gestationAge', condition='in between', upper_bound=43.5, lower_bound=42.5)
S1.get_column_stats('gestationAge', float_round2Int=True)


k = wuml.KDE(S1)
D1 = k.generate_samples(100)

D_35_37 = D1.get_all_samples_based_on_condition_of_column('gestationAge', condition='in between', upper_bound=43.5, lower_bound=42.5)
newData.append_rows(D_35_37)
newData.get_column_stats('gestationAge', float_round2Int=True)


gAge = newData.get_columns('gestationAge', return_type='ndarray')
clasLabel = (gAge < 37).astype(int)
newData.replace_label_with_column('gestationAge')
newData.Data_preprocess(preprocess_data='center and scale')
newData.append_columns(clasLabel, column_names='preterm_best')
newData.mean_of_each_column()

newData.to_csv('../ikids_prot/data/D13.csv', add_row_indices=False, include_column_names=True, float_format='%.4f')









##	Use flow to model the fewer feature original case
#data = wuml.wData(xpath='../ikids_prot/data/fewer_feature_original.csv', batch_size=32, first_row_is_label=True, columns_to_ignore='preterm_best')
#Pᵳ = wuml.flow(data, max_epochs=4000, network_width=1024)
#Pᵳ.save('all_cases_small.model') 



##	Use flow to model the positive cases
#data = wuml.wData(xpath='../ikids_prot/data/D10.csv', batch_size=32, first_row_is_label=True, columns_to_ignore=['preterm_best'])
#Pᵳ = wuml.flow(data, max_epochs=6000, network_width=1024)
#Pᵳ.save('all_cases.model') 


##	Use flow to model the positive cases
#data = wuml.wData(xpath='../ikids_prot/data/D10_positive_cases.csv', batch_size=32, first_row_is_label=True)
#Pᵳ = wuml.flow(data, max_epochs=6000, network_width=1024)
#Pᵳ.save('positive2.model') 
#
##	Use flow to model the negative cases
#data = wuml.wData(xpath='../ikids_prot/data/D10_negative_cases.csv', batch_size=32, first_row_is_label=True)
#Pᵳ = wuml.flow(data, max_epochs=6000, network_width=1024)
#Pᵳ.save('negative2.model') 




#Pᵳ = wuml.flow(data, load_model_path='flow.model')
#samplesᵳ = Pᵳ.generate_samples(10)
#probᵳ = Pᵳ(data)


##	after saving the model, reload this model and generate samples based on it
#Pᵳ = wuml.flow(data, load_model_path='flow.model')
#samplesᵳ = Pᵳ.generate_samples(2000)
#probᵳ = Pᵳ(data)
#samplesᵳ.plot_2_columns_as_scatter(0, 1)
#
#
#
##	compare flow to KDE models
#Pᴋ = wuml.KDE(data)
#probᴋ = Pᴋ(data)
#samplesᴋ = Pᴋ.generate_samples(2000)
#samplesᴋ.plot_2_columns_as_scatter(0, 1)
#
#S = scatter(data.X[:,0], data.X[:,1], title='Original data', subplot=131, figsize=(10,4))
#scatter(samplesᴋ.X[:,0], samplesᴋ.X[:,1], title='KDE Generated', subplot=132)
#scatter(samplesᵳ.X[:,0], samplesᵳ.X[:,1], title='Flow Generated', subplot=133)
#S.show()
