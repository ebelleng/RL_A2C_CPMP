from greedy import *
from enviroment import * 

def main():
    rows, columns = (7,7)
    critic_file = 'critic_model.h5'
    #model = load_critic_model(rows,columns, critic_file)
    
    
    env = Enviroment(rows, columns)

    layout = env.create_env(N=20)
    
    actions = greedy_solve(layout) 
    
    cur_state = layout.stacks
    
    for action in actions:
        print(f'Action {action} :')
        new_state, reward, done = env.step(action)
        cur_state = new_state
        env.show_state(cur_state)
        print(cur_state)
    
    env.show_state(cur_state)

if __name__ == "__main__":
    main()
    