'''
Created on 08-Dec-2022

@author: EZIGO
'''
import numpy as np

class Ensemble_Classifier:
    def __init__(self,alpha,beta,gamma,delta,theta,rf,gb,xgb,adb,model):
        self.__alpha=alpha
        self.__beta=beta
        self.__gamma=gamma
        self.__delta=delta
        self.__theta=theta
        self.__rf=rf
        self.__gb=gb 
        self.__xgb=xgb
        self.__adb=adb
        self.__ANN=model
        
    def get_alpha(self):
        return self.__alpha
    def get_beta(self):
        return self.__beta
    def get_gamma(self):
        return self.__gamma
    def get_delta(self):
        return self.__delta
    def get_theta(self):
        return self.__theta
    
    def predict(self,X_test,BS):
        alpha = self.get_alpha()
        beta = self.get_beta()
        gamma = self.get_gamma()
        delta = self.get_delta()
        theta = self.get_theta()
        # Random forest classifier
        rf = self.__rf
        s1 = rf.predict_proba(X_test)

        # GradientBoosting Classifier
        gb = self.__gb
        s2 = gb.predict_proba(X_test)

        # XgBoost classifier
        xgb = self.__xgb
        s3 = xgb.predict_proba(X_test)

        # Adaboost Classifier
        adb = self.__adb
        s4 = adb.predict_proba(X_test)

        # ANN Classifier
        X_test=X_test.values
        X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
        model = self.__ANN
        s5 = model.predict(X_test,batch_size = BS)
        
        y_pred_proba=(alpha*s1)+(beta*s2)+(gamma*s3)+(delta*s4)+(theta*s5)
        return y_pred_proba