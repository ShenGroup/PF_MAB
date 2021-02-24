# -*- coding: utf-8 -*-


import numpy as np

class server(object):
    def __init__(self,
                 narms,
                 nclients):
        self.M = nclients
        self.K = narms
        
        self.local_means = np.zeros([self.M,self.K])
        self.global_means = np.zeros(self.K)
        self.global_set = set()
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
    
    def global_set_update(self):
        return self.global_set
        