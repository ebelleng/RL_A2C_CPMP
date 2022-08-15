from transform_state import *
import copy
from greedy import greedy_solve
import numpy as np
import random
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

def generate_ann_state_lm(yard, H, last_move=None):
    n_rows=len(yard)
    yard=copy.deepcopy(yard)
    max_item = max(set().union(*yard))
    #max_item = max([max(stack) for stack in yard])
    stackValues=getStackValues(yard) #well-placed
    stacksLen = getStackLen(yard)
    topStacks = getTopStacks(yard,max_item)
    yard=compactState(yard)
    yard=elevateState(yard,H, max_item)
    yard=flattenState(yard)
    yard=normalize(yard,max_item)
    state=yard
    state.shape=(n_rows,H)
    state=np.lib.pad(state, ((0,0),(0,2)), 'constant', constant_values=(0))
    if last_move is not None:
      state[last_move[0]][H]=1.0
      state[last_move[1]][H+1]=1.0

    state.shape=((n_rows+2)*H)

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

    return Layout(stacks,H)

def InputEncoder(X: [[[]]], rows: int, columns: int) -> []:
  #X = X
  X_input_array = []
  for i in range(0, columns):
    X_input = X[0:,i]
    X_input.shape = (X.shape[0], rows+2, 1)
    X_input_array.append(X_input)
  return X_input_array

def generate_data_greedy(n=10000, model_type="value", columns=7, rows=7, Nmin=20, Nmax=35):
  states=[]; moves=[]; costs=[]; prev_move=[]
  while len(states)<n:
    N=random.randint(Nmin, Nmax)
    layout = generate_random_layout(rows,columns,N)
    layout_ = copy.deepcopy(layout)
    ret = greedy_solve(layout)
    if ret is None: continue

    for m in layout.moves:
      states.append(copy.deepcopy(layout_.stacks))
      costs.append(layout.steps-layout_.steps) #cantidad de pasos para resolver el problema a partir de layout_
      lb = compute_unsorted_elements(layout_.stacks)
      moves.append([m[0],m[1]])
      if layout_.steps>0: prev_move.append(layout_.moves[-1])
      else: prev_move.append(None)

      layout_.move(m[0],m[1])
      if len(states)==n or lb==(layout.steps-layout_.steps): break;

  df_numpy=[]
  for i in range(len(states)):
    lb = compute_unsorted_elements(states[i]) #cantidad de elementos mal ubicados en layout actual (lower bound)
    st = generate_ann_state_lm(states[i],rows,moves[i])
    print(st.shape)
    df_numpy.append(np.concatenate((st,np.array([costs[i]-lb]+moves[i]))))
  df_numpy = np.array(df_numpy)

  X = df_numpy[:,:(rows+2)*columns]
  y = df_numpy[:,-3] # costs[i]-lb
  
  # reshape para X
  X = np.expand_dims(X, axis=2)
  print(X.shape)
  X.shape = (X.shape[0], columns, rows+2)
  print(X.shape)
  return X,y

def main():
  X,y = generate_data_greedy(1)
  print(X)

main()