import random

class Enviroment():
    def __init__(self, S, H):
        self.S = S  #number stacks
        self.H = H  #number tiers
        
    def create_env(self, N):
        stacks = []
        for i in range(self.S):
            stacks.append([])
        
        for j in range(N):
            s=random.randint(0,self.S-1);
            while len(stacks[s])==self.H: s=s=random.randint(0,self.S-1);
            stacks[s].append(random.randint(1,N));
        
        self.layout = stacks
        print(stacks)
        return stacks

    def step(self, action):
        new_state = 1
        reward = 1
        done = 1
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

        return

    # ========================================================================= #
    #                                Debug                                      #
    # ========================================================================= #
    
    def show_state(self, cur_state):
        #print( [ fila + [0]*(H-len(fila)) for fila in layout ] )

        lay = [ fila + [0]*(self.H-len(fila)) for fila in cur_state ]

        for i in range(self.H-1, -1, -1):
            for j in range(len(lay)):
                print(lay[j][i], end=' ')
            print()