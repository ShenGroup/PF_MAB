# -*- coding: utf-8 -*-


import numpy as np

class imp_client(object):
    def __init__(self,
                 index,
                 thorizon,
                 narms,
                 nclients,
                 palpha,
                 fp):
        self.T = thorizon
        self.id = index
        self.K = narms
        self.M = nclients
        self.alpha = palpha
        self.fp = fp
        
        self.p = 1
        self.local_set = set(np.arange(self.K))
        self.global_set = set(np.arange(self.K))
        self.local_mean = np.zeros(self.K)
        self.global_mean = np.zeros(self.K)
        self.mixed_mean = np.zeros(self.K)
        
        self.local_delta = np.ones(self.K)
        self.global_delta = np.ones([self.M,self.K])
        
        self.reward = np.zeros(self.K)
        self.pull = np.zeros(self.K)
        self.p_length = self.fp(self.p)
        self.Fp = 0
        
        self.farm = 0
        self.farm_pull = np.zeros(self.K)
        self.garm = 0
        self.garm_pull = np.zeros(self.K)
        self.imp_factor = (self.alpha*np.sqrt(self.local_delta)+(1-self.alpha)/self.M*np.sum(np.sqrt(self.global_delta),axis=0))/np.sqrt(self.local_delta)
        self.farm_pull_bound = np.ceil((1-self.alpha)*self.p_length*self.imp_factor)
        self.garm_pull_bound = np.ceil(self.alpha*self.M*self.p_length*self.imp_factor)
        
        self.F = -1
        self.l_exploration = False
        self.g_exploration = False
    
    def play(self):
        if self.farm < len(self.global_set):
            play = list(self.global_set)[int(self.farm)]
            self.farm_pull[play] += 1
            if self.farm_pull[play] >= self.farm_pull_bound[play]:
                self.farm += 1
                
        elif self.garm < len(self.local_set):
            play = list(self.local_set)[int(self.garm)]
            self.garm_pull[play] += 1
            if self.garm_pull[play] >= self.garm_pull_bound[play]:
                self.garm += 1
            
        else: #exploitation phase
            if self.l_exploration is True:
                play = self.F
            else:
                play = np.argmax(self.alpha*self.local_mean+(1-self.alpha)*self.global_mean)
            
        return play
    
    def reward_update(self,play,obs):
        self.reward[play] += obs
        self.pull[play] += 1
        
    def local_mean_update(self):
        #print('global',self.fphase,np.ceil((1-self.alpha)*self.p_length)*len(self.global_set))
        #print('local',self.gphase,np.ceil(self.M*self.alpha*self.p_length)*len(self.local_set))
        if self.g_exploration is False and self.farm>= len(self.global_set) and self.garm>= len(self.local_set):
            self.local_mean = self.reward/self.pull
            #print("local_mean",self.local_mean, "phase", self.p)
            return True, self.local_mean
        else:
            return False, 0
        
    def global_mean_update(self,global_stat):
        self.global_mean = global_stat
        self.mixed_mean = self.alpha*self.local_mean+(1-self.alpha)*self.global_mean
        
    def local_set_update(self):
        Ep = set()
        self.Fp += self.p_length
        conf_bound = np.sqrt(np.log(self.T)/(self.M*self.Fp))
        for i in list(self.local_set):
            if self.mixed_mean[i]+conf_bound < max(self.mixed_mean-conf_bound):
                Ep.add(i)
        self.local_set = self.local_set-Ep
        
        if len(self.local_set) == 1 and self.l_exploration is False:
            self.l_exploration = True
            self.F = list(self.local_set)[0]
            #print("player", self.id,  " fixate",self.F)
            self.local_set = set()
        #print("player", self.id, " local-set:",self.local_set)    
        return self.local_set
        
    def local_delta_update(self):
        self.local_delta = np.minimum(np.ones(self.K)*np.max(self.mixed_mean)-self.mixed_mean+2*np.sqrt(np.log(self.T)/(self.M*self.Fp)),np.ones(self.K))
        #self.local_delta = np.minimum(np.ones(self.K)*np.max(self.mixed_mean)-self.mixed_mean+2*np.sqrt(np.log(self.T)/(self.M*self.Fp)),np.ones(self.K))
        #self.local_delta = np.ones(self.K)
        return self.local_delta
        
    def global_set_update(self,global_set):
        self.global_set = global_set
        
    def global_delta_update(self,global_delta):
        self.global_delta = global_delta
        self.p +=1
        self.p_length = self.fp(self.p)
        #print("p-length",self.p_length)
        self.farm = 0
        self.farm_pull = np.zeros(self.K)
        self.garm = 0
        self.garm_pull = np.zeros(self.K)
        self.imp_factor = (self.alpha*np.sqrt(self.local_delta)+(1-self.alpha)/self.M*np.sum(np.sqrt(self.global_delta),axis=0))/np.sqrt(self.local_delta)
        #print("imp factor:", self.imp_factor)
        self.farm_pull_bound = np.ceil((1-self.alpha)*self.p_length*self.imp_factor)
        self.garm_pull_bound = np.ceil(self.alpha*self.M*self.p_length*self.imp_factor)
        
        self.g_exploration = (len(self.global_set)==0)
        
            
            
    
            