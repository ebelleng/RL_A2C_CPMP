from sklearn.model_selection import train_test_split
from enviroment import Enviroment
from transform_state import *
import copy
from greedy import greedy_solve
import numpy as np
import random
import pandas as pd
from layout import *


def compute_unsorted_elements(stacks):
    total = 0
    for stack in stacks:
        if len(stack) == 0:
            continue
        sorted_elements = 1
        while(sorted_elements < len(stack) and stack[sorted_elements] <= stack[sorted_elements-1]):
            sorted_elements += 1

        total += len(stack)-sorted_elements
    return total

def InputEncoder(X: [[[]]], rows: int, columns: int) -> []:
    X_input_array = []
    for i in range(0, columns):
        X_input = X[0:,i]
        X_input.shape = (X.shape[0], rows+2, 1)
        X_input_array.append(X_input)
    return X_input_array

def generate_ann_state_lm(yard, H, last_move=None):
    n_rows      = len(yard)
    yard        = copy.deepcopy(yard)
    max_item    = max(set().union(*yard))
    #max_item   = max([max(stack) for stack in yard])
    stackValues = getStackValues(yard)  # well-placed
    stacksLen   = getStackLen(yard)
    topStacks   = getTopStacks(yard, max_item)
    yard        = compactState(yard)
    yard        = elevateState(yard, H, max_item)
    yard        = flattenState(yard)
    yard        = normalize(yard, max_item)
    state       = yard
    state.shape = (n_rows, H)
    state       = np.lib.pad(state, ((0, 0), (0, 2)),
                       'constant', constant_values=(0))
    if last_move is not None:
        state[last_move[0]][H] = 1.0
        state[last_move[1]][H+1] = 1.0

    state.shape = ((n_rows+2)*H)

    return state

def generate_random_layout(S, H, N):
    stacks = []
    for i in range(S):
        stacks.append([])

    for j in range(N):
        s = random.randint(0, S-1)
        while len(stacks[s]) == H:
            s = s = random.randint(0, S-1)
        stacks[s].append(random.randint(1, N))

    return Layout(stacks, H)

def generate_data_a2c(n=1000,actor_critic=None,rows=7,columns=7,a2c_step=1):
    gamma=1
    env = Enviroment(rows,columns)
    data = {
        'states'     : [],
        'advantages' : [],
        'rewards'    : [],
        'actions'    : []
        }

    for i in range(n):
        layout = env.create_env((rows*columns) // 2) ; current_state = layout.stacks
        states_i,actions_i,advantages_i,rewards_i,done = actor_critic.solve(layout,greedy_solve,n_pasos=a2c_step)
        if done: 
            states     += states_i
            actions    += actions_i
            advantages += advantages_i
            rewards    += rewards_i
            continue

        greedy_actions = greedy_solve(copy.deepcopy(layout))
        reward_acum = 0
        for action in greedy_actions:
            new_state, _, done = env.step(layout, action)
            actions_i.append(action)

            # Calculamos malas posiciones estado actual y siguiente
            BP_i  = len( env.get_bad_positions(current_state))
            BP_i_ = len( env.get_bad_positions(new_state))

            # Calculamos recompensa inmediata
            if done: reward = -len(greedy_actions) - 1
            else   : reward = (BP_i - BP_i_) - 1
            
            #print(f'# {done}-> Rwds_acum: {reward_acum} - Rwds: {reward}')
            reward_acum += reward

            Vs = actor_critic.critic_predict(current_state)
            Vs_ = actor_critic.critic_predict(new_state)    # Valor estado siguiente
            
            current_state = new_state
            states_i.append(copy.deepcopy(current_state))
            rewards_i.append(reward_acum)

            advantages_i.append(reward + gamma * Vs_ - Vs)
        
        data['states'].append(states_i)
        data['actions'].append(actions_i)
        data['advantages'].append(advantages_i)
        data['rewards'].append(rewards_i)

    df = pd.DataFrame(data, columns=['states','actions','advantages','rewards'])
    df.to_csv('data\\data.csv', index=None)
 
    X = df[['states','advantages']]
    y = df[['actions','rewards']]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=42)
    X_train.to_csv('data\\x_train.csv',index=None)
    X_test.to_csv('data\\x_test.csv',index=None)
    y_train.to_csv('data\\y_train.csv',index=None)
    y_test.to_csv('data\\y_test.csv',index=None)

    return X_train,X_test,y_train,y_test

def generate_data_greedy(n=10000, columns=7, rows=7, Nmin=20, Nmax=35):
    states = []
    moves = []
    costs = []
    prev_move = []
    while len(states) < n:
        N = random.randint(Nmin, Nmax)
        layout = generate_random_layout(rows, columns, N)
        layout_ = copy.deepcopy(layout)
        ret = greedy_solve(layout)
        if ret is None:
            continue

        for m in layout.moves:
            states.append(copy.deepcopy(layout_.stacks))
            # cantidad de pasos para resolver el problema a partir de layout_
            costs.append(layout.steps-layout_.steps)
            lb = compute_unsorted_elements(layout_.stacks)
            moves.append([m[0], m[1]])
            if layout_.steps > 0:
                prev_move.append(layout_.moves[-1])
            else:
                prev_move.append(None)

            layout_.move(m[0], m[1])
            if len(states) == n or lb == (layout.steps-layout_.steps):
                break


    ## ACTOR DATA
    index_list = [(i,j) for i in range(rows) for j in range(columns) if i != j]
    df_numpy = []
    for i in range(len(states)):
        st =  (states[i], rows)
        index_move = index_list.index( tuple(moves[i]) ) #[(0,1), (0,2)...]
        
        # Crear vector de -1, con max valor 0 -> [-1,-1,...,0,-1]
        # vector = np.full((len(index_list),), -1)
        # vector[index_move] = 0
        df_numpy.append(np.concatenate((st, np.array([index_move]))))
    
    df_numpy = np.array(df_numpy)
    
    X = df_numpy[:, :(rows)*columns]
    y = df_numpy[:, -1]

    # reshape para X
    X = np.expand_dims(X, axis=2)
    X.shape = (X.shape[0], columns, rows)
    X_actor = X
    y_actor = y
    
    ## CRITIC DATA
    df_numpy = []
    for i in range(len(states)):
        # cantidad de elementos mal ubicados en layout actual (lower bound)
        lb = compute_unsorted_elements(states[i])
        st = generate_ann_state_lm(states[i],rows,moves[i])
        df_numpy.append(np.concatenate((st,np.array(moves[i]+[costs[i]-lb]))))
            
    df_numpy = np.array(df_numpy)

    X = df_numpy[:,:(rows+2)*columns]
    y = df_numpy[:,-1] # costs[i]-lb
    
    # reshape para X
    X = np.expand_dims(X, axis=2)
    X.shape = (X.shape[0], columns, rows+2)
    X_critic = X
    y_critic = y

    return X_actor, y_actor, X_critic, y_critic


def main():
    X_actor, y_actor, X_critic, y_critic = generate_data_greedy(1)

    print(X_actor)
    print(y_actor)
    print(X_critic.shape)
    print(y_critic.shape)


if __name__ == "__main__":
    # main('train')
    main()
