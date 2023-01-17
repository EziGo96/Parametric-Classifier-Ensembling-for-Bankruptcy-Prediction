'''
Created on 08-Dec-2022

@author: EZIGO
'''
from config import alpha_list,beta_list,gamma_list,delta_list,theta_list
import numpy as np

params=[]
for alpha in alpha_list:
    for beta in beta_list:
        for gamma in gamma_list:
            for delta in delta_list:
                for theta in theta_list:
                    if sum([round(alpha, 2),round(beta, 2),round(gamma, 2),round(delta, 2),round(theta, 2)])==1.00:
                        params.append([alpha,beta,gamma,delta,theta])

print(len(params))
                        