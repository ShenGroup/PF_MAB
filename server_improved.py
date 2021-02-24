# -*- coding: utf-8 -*-

import numpy as np

class imp_server(object):
    def __init__(self,
                 narms,
                 nclients):
        self.M = nclients
        self.K = narms
        
        self.local_means = np.zeros([self.M,self.K])
        self.global_means = np.zeros(self.K)
        self.global_set = set()
        self.global_delta = np.ones([self.M, self.K])
        
        self.p = 1
        self.c_local_stat = np.zeros(self.M)
        
    def local_mean_update(self,i,local_stat):
        self.local_means[i] = local_stat
        self.c_local_stat[i] = 1
        
    def global_mean_update(self):
        self.global_set = set()
        if sum(self.c_local_stat) >= self.M:
            self.global_means = np.sum(self.local_means, axis=0)/self.M
            self.c_local_stat = np.zeros(self.M)
            return True, self.global_means
        else:
            return False, 0
       
    def local_set_update(self,i,local_set):
        self.global_set = self.global_set|local_set
    
    def local_delta_update(self,i,local_delta):
        self.global_delta[i] = local_delta
    
    def global_set_update(self):
        return self.global_set
    
    def global_delta_update(self):
        #print("global delta:", self.global_delta)
        return self.global_delta