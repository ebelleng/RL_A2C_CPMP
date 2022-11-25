from actor_critic import ActorCritic
from greedy import *
from data import generate_data_a2c
from enviroment import Enviroment
import tqdm
import collections
import gym
import numpy as np
import statistics
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple

def main(epochs=10):
    rows,columns = 7,7
    env          = Enviroment(rows,columns)

    num_actions      = (rows-1)*columns, 
    num_hidden_units = 128
    actor_critic     = ActorCritic(num_actions,num_hidden_units)
    
    max_episodes = 1
    max_steps_per_episode = 30
    min_episodes_criterion = 10

    reward_threshold = 20
    running_reward = 0

    # Keep last episodes reward
    episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

    with tqdm.trange(max_episodes) as t:
        for i in range(max_episodes):
            initial_state = env.create_env((rows*columns) // 2)
            print(initial_state.stacks)
            episode_reward = actor_critic.train_step(initial_state, actor_critic, max_steps_per_episode)
            
            episodes_reward.append(episode_reward)
            running_reward = statistics.mean(episodes_reward)
            
            t.set_description(f'Episode {i}')
            t.set_postfix(
                episode_reward=episode_reward, running_reward=running_reward)

            # Show average episode reward every 10 episodes
            if i % 10 == 0:
                pass # print(f'Episode {i}: average reward: {avg_reward}')
                
            if running_reward > reward_threshold and i >= min_episodes_criterion:  
                break
            
        np.savetxt('episodes_rewards.csv', episodes_reward)
    
    # # Epoca
    # for epoch in range(epochs): # Grafica
    #     print(epoch)
    #     X_train,X_test,y_train,y_test  = generate_data_a2c(10,actor_critic,rows,columns)
    #     actor_critic.fit(X_train,y_train,verbose=True,save_files=True)

if __name__ == "__main__":
    main()
