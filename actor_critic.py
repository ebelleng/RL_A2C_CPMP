import numpy as np
import random

from keras.models import Model
from keras.layers import Input, Dense
from keras.layers.merge import Add
from tensorflow.keras.optimizers import Adam


class ActorCritic:
    def __init__(self, env):
        self.env = env

        # ===================================================================== #
        #                               Actor Model                             #
        # Chain rule: find the gradient of chaging the actor network params in  #
        # getting closest to the final value network predictions, i.e. de/dA    #
        # Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
        # ===================================================================== #

        self.memory = []

        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        # ===================================================================== #
        #                              Critic Model                             #
        # ===================================================================== #
        
        self.critic_state_input, self.critic_action_input, \
			self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()
   
    # ========================================================================= #
    #                              Model Definitions                            #
    # ========================================================================= #

    def create_actor_model(self):
        state_input = Input(shape=self.env.get_shape())
        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        output = Dense(2, activation='relu')(h3) #action

        model = Model(state_input, output)
        adam  = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)

        return state_input, model

    def create_critic_model(self):
        state_input = Input(shape=self.env.get_shape())
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48)(state_h1)

        action_input = Input(shape=(2,))
        action_h1    = Dense(48)(action_input)

        merged    = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1) #dimension output

        model = Model([state_input, action_input], output)

        adam  = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)

        return state_input, action_input, model

    # ========================================================================= #
    #                               Model Training                              #
    # ========================================================================= #

    def remember(self, cur_state, action, reward, new_state, done):
        print(action)
        self.memory.append([cur_state, action, reward, new_state, done])

    def _train_actor(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, _ = sample
            predicted_action = self.actor_model.predict(cur_state)


    def _train_critic(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            if not done:
                target_action = self.target_actor_model.predict(new_state)
                future_reward = self.target_critic_model.predict(
					[new_state, target_action])[0][0]
                reward += self.gamma * future_reward
            self.critic_model.fit([cur_state, action], reward, verbose=0)

    def train(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        rewards = []
        samples = random.sample(self.memory, batch_size)
        self._train_critic(samples)
        self._train_actor(samples)

    # ========================================================================= #
    #                         Target Model Updating                             #
    # ========================================================================= #

    def _update_actor_target(self):
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_critic_model.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        self.target_critic_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.critic_target_model.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]
        self.critic_target_model.set_weights(critic_target_weights)

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()

    # ========================================================================= #
    #                              Model Predictions                            #
    # ========================================================================= #

    def act(self, cur_state, possible_actions):
        # Encargada de elegir la accion segun su politica
        # Fase 1: greedy recorrer secuencia del greedy (retorna lista acciones)
        # Fase 2: bellman

        # actor evlua accciones segun recom acum.
        # 1. Predecir origen (input: cur_state)
        # 2. Predecir detino (input: origen, cur_state)
        origen = 0
        destino = 0
        return (origen, destino)
    
    # ========================================================================= #
    #                              Model Predictions                            #
    # ========================================================================= #

    def metrics(self):
        return 
        
    # ========================================================================= #
    #                                Debug                                      #
    # ========================================================================= #
    
    def show(self):
        print("Hello world")

    def get_memory(self):
        return self.memory