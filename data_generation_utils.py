
## Data generation
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import pickle as pkl

def sin_prob(x):
    '''
 This function is the sin function
 
 Input is x: the covariate
 Output is (np.sin(x*2)+3)/8: the sin value 
    '''
    return (np.sin(x*2)+3)/8



def generate_t_matrix_censored(age,p):
    '''
  This function is to get the transition with two absorbing states death and censoring.
  
  Input is age: covarate;
           p: censoring probability

  Output is t_matrix: the transition matrix
    '''

    p_sub = p/3
    t_matrix = np.array([[sin_prob(age)-p_sub,((1-sin_prob(age))/2)-p_sub,((1-sin_prob(age))/2)-p_sub,0,p], 
                         [(sin_prob(age)/2)-p_sub,1-sin_prob(age)-p_sub,(sin_prob(age)/2)-p_sub,0,p], 
                         [sin_prob(age)*(2/3)-p_sub,1-sin_prob(age)-p_sub,sin_prob(age)*(1/3)-p_sub,0,p], 
                         [0,0,0,1,0], 
                         [0,0,0,0,1]])
    return t_matrix


def generate_sequence_drift_censored(t_matrix,
                            seq_len, 
                            rules,
                            s0,
                            return_T = False):
    '''
    This function is to generate sequences with censoring included
    
Input is t_matrix: the transition matrix;
          seq_len: length of the sequence;
             rules: stopping criteria;
                s0: probability vector of the initial state
          return_T: if the function returns the event and time

Output is seq: the sequence 
If return_T == FALSE, output is seq: the sequence; E: events; T: time
    '''
    
    n_states = t_matrix.shape[0]
    cur_state = np.random.choice(range(len(s0)),p=s0)
    seq = [cur_state]
    rule_met = None
    E,T = 0,seq_len-1
    for i in range(1,seq_len):
        cur_state = np.random.choice(range(n_states), p=t_matrix[cur_state,:])
        seq.append(cur_state)
        for rule in rules:
            if seq[-len(rule):] == rule:
                seq[-1] = n_states-2
                cur_state = n_states-2
                T = i
                E = 1
        if cur_state == (n_states-1) and T == seq_len-1: #its been censored
            T = i
            E = 0
    if return_T:
        return seq,E,T
    else:
        return seq
   




def generate_dataset_censored(rules, seq_len, s0, n = 100000, p=.1):

    '''
    This function is to generate sequence data
 
    Input is rules: stopping criteria;
        seq_len: the length of the sequence;
             s0: probability vector of the initial state
              n: sample size
              p: censoring probability

    Output is seqs: the sequence generated
             Es: events
             Ts: time
           ages: covariates 
    '''

    ages = np.random.uniform(low=0,high=5,size=n)
    Es,Ts = [],[]
    seqs = []
    for age in tqdm_notebook(ages):
        t_matrix = generate_t_matrix_censored(age,p)
        seq, E, T = generate_sequence_drift_censored(t_matrix,seq_len,rules,s0,return_T=True)
        seqs.append(seq)
        Es.append(E)
        Ts.append(T)
    return np.array(seqs),Es,Ts,ages


# Set the initial state probability vector s0
s0 = [1/3,1/3,1/3]
# Set the stopping criteria
rules = [[0,0,0],[1,1,1],[2,2,2]]

## Functions for ground truth and data loading

def get_T_matrix(t_matrix, rules):
    '''
     This function is to obtain the expanded transition matrix from the data generation transition matrix
    
 Input is t_matrix: data generation transition matrix
             rules: stopping criteria

 Output is T_matrix:expanded transition matrix;
        T_matrix_df: the columns (I changed the code and did not use this);
         state_dict: the code for each states
    '''


    dim_expanded = t_matrix.shape[0] + len(rules)
    # censored probability
    p = t_matrix[0, t_matrix.shape[0] - 1]
    
     # Set censored column
    T_matrix = np.zeros((dim_expanded, dim_expanded))
    T_matrix[0:(dim_expanded - 2), dim_expanded - 2] = p
    # T_matrix[0:(dim_expanded - 1), dim_expanded - 1] = p
    # Set death column
    state_prob = t_matrix.diagonal()
    # probability of death equal to the diagnal matrix of the transition matrix
    state_num = len(state_prob) - 2
   # T_matrix[(t_matrix.shape[0] - 2):(t_matrix.shape[0] - 2 + len(rules)), dim_expanded - 2] = state_prob[range(state_num)]
    T_matrix[(t_matrix.shape[0] - 2):(t_matrix.shape[0] - 2 + len(rules)), dim_expanded - 1] = state_prob[range(state_num)] 

    # set basic state transition probabilities
    T_matrix[0:state_num, 0:state_num] = t_matrix[0:state_num, 0:state_num]

    # From state to transition rules
    T_matrix[0:state_num,state_num:state_num + len(rules)] = np.diag(np.diag(t_matrix[0:state_num, 0:state_num]))  
    
    # Set diagonal probabilities    
    for i in range(dim_expanded):
        T_matrix[i, i] = 0
        if i >= dim_expanded - 2:
            T_matrix[i, i] = 1
    
    # From transition rules to state
    T_matrix[state_num:state_num + len(rules), 0:state_num] = T_matrix[0:state_num, 0:state_num]
     
    T_matrix_df = dim_expanded 
    state_dict = {'[0]': 0,
                  '[1]': 1,
                  '[2]': 2,
                  '[0, 0]': 3,
                  '[1, 1]': 4,
                  '[2, 2]': 5,
                  '[4]': 6,
                  '[3]': 7
                  
}
 


    return T_matrix,T_matrix_df,state_dict



def get_ground_truth(ages,seq_len=20,starting_state = None):
# This function is to get the ground truth survival probabilities for each state
# Input is ages: covariates
#       seq_len: the length of the sequence
#starting_state: initial state
# 
# Output is arrarys of survival probabilities for each state for every covariate entered
    def k_mult_probs(s,t_matrix,k):
        # This function is to calculate the probability to the next state according to the previous state and transition matrix
        return s@(np.linalg.matrix_power(t_matrix,k))
    
    rules = [[0,0,0],[1,1,1],[2,2,2]]
    if starting_state != None:
        s_0 = np.array([0,0,0,0,0,0,0,0])
        s_0[starting_state] = 1
        extra_y = False 
    else:
        s_0 = np.array([1/3,1/3,1/3,0,0,0,0,0])
        extra_y = True #need to add an extra 1
        
    # censor_state = 4
    ground_truth = []
    n = 100000
    
    for i in ages:
        t_matrix = generate_t_matrix_censored(i,.1)
        T_matrix,T_matrix_df,state_dict = get_T_matrix(t_matrix,rules)
        # censor_index = (T_matrix_df.columns == '[{}]'.format(censor_state)).argmax()
        censor_index = 6
        y = [1.]
        if extra_y == True:
            y.append(1.)
            
        # calculate the survival probabilities    
        survival_prob = 1.
        n_k = n
        for k in range(1,seq_len):
            k_probs = k_mult_probs(s_0,T_matrix,k)
            k_prev = k_mult_probs(s_0,T_matrix,k-1)

            d_k = (k_probs[-1]-k_prev[-1])*n
            c_k = (k_probs[censor_index]-k_prev[censor_index])*n
            #c_k = (k_probs[censor_index] - k_prev[censor_index])*n
            survival_prob *= (1 - (d_k)/n_k)

            n_k = n_k - d_k - c_k
            y.append(survival_prob)
        ground_truth.append(y)
    return np.array(ground_truth).squeeze(), state_dict


# functions to unpickle data and puyt them in tensors
class survival_dataset_cont(Dataset):

    def __init__(self, file, SOS=None, normed=False):
        self.normed = normed
        data = pkl.load(file)
        x,Es,Ts,ages = data

        if SOS != None:
            x = np.pad(x,((0,0),(1,0)),mode='constant',constant_values=SOS)
        self.len = x.shape[0]
        self.x = torch.Tensor(x[:,:-1]).long()
        self.y = torch.Tensor(x[:,1:]).long()
        self.Es = torch.Tensor(Es)
        self.Ts = torch.Tensor(Ts)
        self.ages = torch.Tensor(ages)
        if normed == True:
            self.ages_normed = (self.ages-self.ages.mean())/self.ages.std()

    def __getitem__(self, index):
        if self.normed:
            return self.x[index], self.y[index], self.ages[index],self.ages_normed[index],self.Es[index], self.Ts[index]
        else:
            return self.x[index], self.y[index], self.ages[index], self.Es[index],self.Ts[index]

    def __len__(self):
        return self.len
