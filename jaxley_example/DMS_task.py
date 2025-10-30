
import numpy as np
class DMS():
    """
    Class for generating delayed match to sample task data
    """
    def __init__(self, task_params):
        self.params = task_params
    
    def gen_batch(self):
        """
        Generate a batch of trials with randomly generated delays and stimulus identities
        Returns:
            Stim: stimulus array [batch_size, duration, 2]
            target: target array [batch_size, duration, 2]
            mask: target array [batch_size, duration, 2]
        """
        n_out=2
        if len(self.params["stim_onset"])>1:
            ramp_up =(np.random.uniform(low = self.params['stim_onset'][0], high = self.params['stim_onset'][1],size=(self.params['batch_size'],)))
            ramp_up = ramp_up.astype(int)
        else:
            ramp_up = [self.params["stim_onset"][0]]*self.params['batch_size']
      
        if len(self.params["delay"])>1:
            delay =(np.random.uniform(low = self.params['delay'][0], high = self.params['delay'][1],size=(self.params['batch_size'],)))
            delay = delay.astype(int)
        else:
            delay = [self.params["delay"][0]]*self.params['batch_size']
      
        sim_time = self.params["stim_len"]*2 + self.params['delay'][-1] + self.params["stim_onset"][-1] + self.params['response_onset'] + self.params["response"]
        stim1_end = np.array(ramp_up) + self.params["stim_len"]
        delay_end = stim1_end + np.array(delay)
        stim2_end = delay_end + self.params["stim_len"]
        response_onset = stim2_end + self.params['response_onset']
        response_end = response_onset+self.params['response']

        stim = np.zeros((self.params['batch_size'], sim_time,n_out))
        target = np.zeros((self.params['batch_size'], sim_time,n_out))
        mask = np.zeros((self.params['batch_size'], sim_time,n_out))

        stim1 = np.random.randint(0,2,self.params['batch_size'])
        stim2 = np.random.randint(0,2,self.params['batch_size'])
        match = (stim1==stim2).astype(int)

        for i in range(self.params['batch_size']):
            stim[i, ramp_up[i]:stim1_end[i],stim1[i]] += self.params['stim_amp']+np.random.randn(self.params['stim_len'])*self.params['stim_noise_sd']
            stim[i, delay_end[i]:stim2_end[i],stim2[i]] += self.params['stim_amp']+np.random.randn(self.params['stim_len'])*self.params['stim_noise_sd']
            target[i,response_onset[i]:,match[i]]=1
            mask[i,response_onset[i]:response_end[i]]=1

        return stim, target, mask
    
    def gen(self):
        """
        Generate four trials, one corresponding to each possible stimulus combination
        Returns:
            Stim: stimulus array [4, duration, 2]
            target: target array [4, duration, 2]
            mask: target array [4, duration, 2]
        """
        n_trials=4
        n_out=2
        if len(self.params["stim_onset"])>1:
            ramp_up =(np.random.uniform(low = self.params['stim_onset'][0], high = self.params['stim_onset'][1],size=(n_trials,)))
            ramp_up = ramp_up.astype(int)
        else:
            ramp_up = [self.params["stim_onset"][0]]*n_trials
      
        if len(self.params["delay"])>1:
            delay =(np.random.uniform(low = self.params['delay'][0], high = self.params['delay'][1],size=(n_trials,)))
            delay = delay.astype(int)
        else:
            delay = [self.params["delay"][0]]*n_trials

      
        sim_time = self.params["stim_len"]*2 + self.params['delay'][-1] + self.params["stim_onset"][-1] + self.params['response_onset'] + self.params["response"]
        stim1_end = np.array(ramp_up) + self.params["stim_len"]
        delay_end = stim1_end + np.array(delay)
        stim2_end = delay_end + self.params["stim_len"]
        response_onset = stim2_end + self.params['response_onset']
        response_end = response_onset+self.params['response']

        stim = np.zeros((n_trials, sim_time,n_out))
        target = np.zeros((n_trials, sim_time,n_out))
        mask = np.zeros((n_trials, sim_time,1))

        stim[0, ramp_up[0]:stim1_end[0],0] += self.params['stim_amp']+np.random.randn(self.params['stim_len'])*self.params['stim_noise_sd']
        stim[1, ramp_up[1]:stim1_end[1],0] += self.params['stim_amp']+np.random.randn(self.params['stim_len'])*self.params['stim_noise_sd']
        stim[2, ramp_up[2]:stim1_end[2],1] += self.params['stim_amp']+np.random.randn(self.params['stim_len'])*self.params['stim_noise_sd']
        stim[3, ramp_up[3]:stim1_end[3],1] += self.params['stim_amp']+np.random.randn(self.params['stim_len'])*self.params['stim_noise_sd']

        stim[0, delay_end[0]:stim2_end[0],0] += self.params['stim_amp']+np.random.randn(self.params['stim_len'])*self.params['stim_noise_sd']
        stim[1, delay_end[1]:stim2_end[1],1] += self.params['stim_amp']+np.random.randn(self.params['stim_len'])*self.params['stim_noise_sd']
        stim[2, delay_end[2]:stim2_end[2],0] += self.params['stim_amp']+np.random.randn(self.params['stim_len'])*self.params['stim_noise_sd']
        stim[3, delay_end[3]:stim2_end[3],1] += self.params['stim_amp']+np.random.randn(self.params['stim_len'])*self.params['stim_noise_sd']

        target[0,response_onset[0]:,1]=1
        target[1,response_onset[1]:,0]=1
        target[2,response_onset[2]:,0]=1
        target[3,response_onset[3]:,1]=1

        for i in range(4):
            mask[i,response_onset[i]:response_end[i]]=1
        return stim, target, mask
   