import copy
import numpy as np

def compactState(yard):
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

def elevateState(yard, h, max_item):
    for stack in yard:
      while len(stack) < h:
        stack.insert(0,1.2*max_item)
    return yard

def flattenState(state):
    flatten = []
    for lista in state:
        for item in lista:
            flatten.append(item)
    return flatten

def normalize(state,max_item):
    return np.array(state)/max_item


def getStackValues(yard): #sorted stacks?
    values = []
    for stack in yard:
        flag = False
        cont = 0
        for i in range(len(stack)):
            if i==0:
                cont += 1
            else:
                if stack[i] <= stack[i-1]:
                    cont += 1
                else: break
        values.append(cont)
    return values

def getStackLen(yard):
    lens = []
    for stack in yard:
        lens.append(len(stack))
    return lens

def getTopStacks(yard,max_item):
    tops = []
    for stack in yard:
        if len(stack) != 0:
            tops.append(stack[len(stack)-1])
        else:
            tops.append(max_item)
    return tops


def generate_ann_state(yard,H):
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

    return state