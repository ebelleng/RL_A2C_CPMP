import copy
import time
import actor_critic as ac
from enviroment import Enviroment
from greedy import greedy_solve

MaxIter = 2000
GreedyIter = 5000
Enviroments = 100

def main():
    S, H = (5,7)
    env = Enviroment(S,H)
    actor_critic = ac.ActorCritic(env)

    # Loop de entrenamiento mediante Greedy
    for _ in range(GreedyIter):
        layout = env.create_env(N=20)
        layout_copy = copy.deepcopy(layout)

        actions = greedy_solve(layout) # Se consigue la lista de acciones que permiten resolver el layout

        initial_state = layout_copy.stacks
        solved_state = layout.stacks
        # env.show_state(initial_state)
        # print(f"actions: {actions}")
        # env.show_state(solved_state)

        env.layout = layout_copy # Se vuelve al estado original del layout antes de ser resuelto

        for action in actions:
            cur_state = copy.deepcopy(env.layout.stacks)
            new_state, reward, done = env.step(action)

            actor_critic.remember(cur_state, action, reward, new_state, done)
    
    actor_critic.train(H)
    actor_critic.memory.clear()

    # actor_critic.print_memory()
    # actor_critic.save_memory()
    # actor_critic.load_memory()

    for _ in range(Enviroments):
        layout = env.create_env(N=20)
        cur_state = layout.stacks
        total_reward = 0

        for epoch in range(MaxIter):
            actions, invalid_actions = env.get_actions(cur_state)
            
            action = actor_critic.act(cur_state, actions)

            if action in invalid_actions:
                new_state, reward, done = env.step(None)
            else:
                new_state, reward, done = env.step(action) 
            
            total_reward += reward
            actor_critic.remember(cur_state, action, reward, new_state, done)

            if done:
                break 

            cur_state = new_state

        print(f"Total reward {total_reward}")
        
        actor_critic.train(H)


    #actor_critic.metrics()

if __name__ == "__main__":
	main()