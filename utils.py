import numpy as np


def _compactState(yard):
    sort = []
    for stack in yard:
      for container in stack:
        if not container in sort:
          sort.append(container)
    sort = sorted(sort)
    maxValue = 0
    for i in range(len(yard)):
      for j in range(len(yard[i])):
        yard[i][j] = sort.index(yard[i][j]) + 1
        if yard[i][j] > maxValue:
          maxValue = yard[i][j]
    return yard

def _elevateState(yard, h, max_item):
    for stack in yard:
      while len(stack) < h:
        stack.insert(0,1.2*max_item)
    return yard

def _flattenState(state):
    flatten = []
    for lista in state:
        for item in lista:
            flatten.append(item)
    return flatten

def _normalize(state,max_item):
    return np.array(state)/max_item

def prepare(state, height):
    max_item = max(set().union(*state))

    state = _compactState(state)
    state = _elevateState(state, height, max_item)
    state = _flattenState(state)
    state = _normalize(state, max_item)

    return state