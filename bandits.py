# -*- coding: utf-8 -*-

import numpy as np
from client import client
from server import server

class PFEDUCB(object):
    def __init__(self,
                 fun_fp,
                 T,
                 means, # M*K
                 alpha,
                 reward='Gaussian'):
        self.M = means.shape[0]
        self.K = means.shape[1]
        self.local_means = means
        self.reward_type = reward
        self.alpha = alpha
        self.T = T
        self.C = 1
        self.comm = 0
        
        self.global_means = np.sum(self.local_means, axis=0)/self.M
        self.clients = [
            client(index = i, thorizon = self.T, narms=self.K, nclients = self.M, palpha = self.alpha, fp=fun_fp) for i in range(self.M)
        ]
        self.server = server(narms=self.K, nclients = self.M)


    def simulate_single_step_rewards(self):
        if self.reward_type == 'Bernoulli':
            return np.random.binomial(1, self.local_means)
        return np.random.normal(self.local_means, 1)

    def simulate_single_step(self, plays):

        local_rews = self.simulate_single_step_rewards()
        global_rews = np.mean(local_rews,axis=0)
        
        local_rewards = np.array([local_rews[i,plays[i]] for i in range(self.M)])
        global_rewards = np.array([global_rews[plays[i]] for i in range(self.M)])
        mixed_rewards = self.alpha*local_rewards+(1-self.alpha)*global_rewards
        #rewards = np.array([self.alpha*local_rews[i,plays[i]]+(1-self.alpha)*global_rews[plays[i]] for i in range(self.M)])

        return local_rewards, global_rewards, mixed_rewards

    def simulate(self):
        """
        Return the vector of regret for each time step until horizon
        """

        local_rewards = []
        global_rewards = []
        mixed_rewards = []
        
        play_history = []
           

        for t in range(self.T):
            plays = np.zeros(self.M)
            
            plays = [(int)(client.play()) for client in self.clients]
            local_rews, global_rews, mixed_rews = self.simulate_single_step(plays)
            #obs, rews = self.simulate_single_step(plays)  # observations of all players
            
            local_rewards.append(np.sum(local_rews))
            global_rewards.append(np.sum(global_rews))
            mixed_rewards.append(np.sum(mixed_rews))
            play_history.append(plays)
            
            #print(plays)
            
            for i in range(self.M):
                self.clients[i].reward_update(plays[i], local_rews[i])  # update strategies of all player

            for i in range(self.M):
                f_local_stat, local_stat = self.clients[i].local_mean_update()
                if f_local_stat is True:
                    self.server.local_mean_update(i,local_stat)
                f_local_stat = False
                    
            f_global_stat, global_stat = self.server.global_mean_update()
            
            if f_global_stat is True:
                for i in range(self.M):
                    self.clients[i].global_mean_update(global_stat)
                    local_set = self.clients[i].local_set_update()
                    self.server.local_set_update(i,local_set)
                
                global_set = self.server.global_set_update()
                #print(t," global-set:",global_set)
                for i in range(self.M):
                    self.clients[i].global_set_update(global_set)
                    local_rewards[-1] -= 2*self.C
                    global_rewards[-1] -= 2*self.C
                    mixed_rewards[-1] -= 2*self.C
                    self.comm += 2*self.C
                    #print(t,"comm")
                f_global_stat = False
            
        top_mixed_means = np.zeros(self.M)
        
        top_arms = np.zeros(self.M, dtype=int)
        sub_gap = np.zeros(self.M)
        for i in range(self.M):
            top_arms[i] = np.argmax(self.alpha*self.local_means[i]+(1-self.alpha)*self.global_means)
            top_mixed_means[i] = self.alpha*self.local_means[i][top_arms[i]]+(1-self.alpha)*self.global_means[top_arms[i]]
            sub_gap[i] = np.sort(self.alpha*self.local_means[i]+(1-self.alpha)*self.global_means)[-2]-top_mixed_means[i]
            #print("player", i, " top_arm:",top_arms[i], " top_mean:", top_mixed_means[i], " gap:", sub_gap[i])
        
        best_case_reward = np.sum(top_mixed_means) * np.arange(1, self.T + 1)
        
        cumulated_local_reward = np.cumsum(local_rewards)
        cumulated_global_reward = np.cumsum(global_rewards)
        cumulated_mixed_reward = np.cumsum(mixed_rewards)

        regret = best_case_reward - cumulated_mixed_reward
#        print(cumulated_mixed_reward)
#        print(best_case_reward)
#        print('regret:',regret[-1], "comm: ", self.comm)
        self.regret = (regret, best_case_reward, cumulated_mixed_reward)
        self.top_mixed_means = top_mixed_means
        return regret, cumulated_local_reward, cumulated_global_reward, cumulated_mixed_reward

    def get_clients(self):
        return self.clients