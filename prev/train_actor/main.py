import copy
import time
import actor_critic as ac
from enviroment import Enviroment


MaxIter = 100
GreedyIter = 1000
Enviroments = 100


def main():
    S, H = (7, 7)
    env = Enviroment(S, H)
    actor_critic = ac.ActorCritic(env)
    
    for _ in range(Enviroments):
        layout = env.create_env(N=20)
        cur_state = layout.stacks
        total_reward = 0

        for epoch in range(MaxIter):
            env.show_state(cur_state)
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

# actor_critic.metrics()

if __name__ == "__main__":
    main()
