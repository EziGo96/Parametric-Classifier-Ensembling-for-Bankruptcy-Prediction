'''
Created on 06-Dec-2022

@author: EZIGO
'''
import numpy as np
Data='Data/'
data_prediction="Company_Bankruptcy_Prediction_Dataset.csv"
data_forecast="bankruptcy_Train.csv"
preprocessed_data_prediction="preprocessed_Company_Bankruptcy_Prediction_Dataset.csv"
preprocessed_data_forecast="preprocessed_bankruptcy_Train.csv"
data11="11th_grade_dropouts.csv"
NUM_EPOCHS = 200
INIT_LR = 0.01
BS = 200 
alpha_list=np.linspace(0,1,11,endpoint=True)
beta_list=np.linspace(0,1,11,endpoint=True)
gamma_list=np.linspace(0,1,11,endpoint=True)
delta_list=np.linspace(0,1,11,endpoint=True)
theta_list=np.linspace(0,1,11,endpoint=True)