import random
from layout import Layout


class Enviroment():
    def __init__(self, S, H):
        self.S = S  #number stacks
        self.H = H  #number tiers
        
    def create_env(self, N):
        stacks = []
        for i in range(self.S):
            stacks.append([])
        
        for j in range(N):
            s=random.randint(0,self.S-1)
            while len(stacks[s])==self.H: s=s=random.randint(0,self.S-1)
            stacks[s].append(random.randint(1,N))
        
        #self.layout = stacks
        self.layout = Layout(stacks, self.H)
        return self.layout

    def step(self, action):
        if action is None:
            return (self.layout.stacks, -3, 0)

        stack_source, stack_destination = action
        self.layout.move(stack_source, stack_destination)

        new_state = self.layout.stacks
        reward = -1
        done = 1

        for i in range(len(self.layout.sorted_stack)):
            if self.layout.sorted_stack[i] is False:
                done = 0
                break
                
        return (new_state, reward, done)

    def get_bad_positions(self, cur_state):
        L = []
        print("rr",cur_state)
        for stack in cur_state:
            if len(stack) > 1:
                stack_piv = stack[0]
                for c in stack[1:]:
                    if c > stack_piv:
                        L.append(c)
        return L

    def get_actions(self, cur_state):
        actions = []
        invalid_actions = []
        for i in range(self.S):
            for j in range(self.S):
                if i != j:
                    if len(cur_state[i]) > 0 and len(cur_state[j]) < self.H:
                        actions.append( (i, j) )
                    else:
                        actions.append( (i, j) )
                        invalid_actions.append( (i, j) )
        return actions, invalid_actions

    def get_shape(self):
        return (self.S, self.H)

    # ========================================================================= #
    #                                Debug                                      #
    # ========================================================================= #
    
    def show_state(self, cur_state):
        #print( [ fila + [0]*(H-len(fila)) for fila in layout ] )

        lay = [ fila + [0]*(self.H-len(fila)) for fila in cur_state ]

        for i in range(self.H-1, -1, -1):
            for j in range(len(lay)):
                print(f'{lay[j][i]:2}', end=' ')
            print()
        print()