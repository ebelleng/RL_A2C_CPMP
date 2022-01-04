import numpy as np
import random

from keras.models import Model
from keras.layers import Input, Dense
from keras.layers.merge import Add
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
import tensorflow_probability as tfp


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

        self.actor_state_input, self.actor_model, self.pi = self.create_actor_model()
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
        n_actions = self.env.get_shape[0] *  (self.env.get_shape[0] - 1)

        state_input = Input(shape=self.env.get_shape())
        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        output = Dense(n_actions, activation='softmax')(h3) #action
        # [(0,1), (0,2), (1,0), (1,2) ]

        model = Model(state_input, output)
        adam  = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)

        return state_input, model, output

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
        self.memory.append([cur_state, action, reward, new_state, done])

    def _train_actor(self, samples): # policy_train
        with tf.GradientTape(persistent=True) as tape:
            for sample in samples:
                cur_state, action, reward, new_state, done = sample

                cur_state = tf.convert_to_tensor([cur_state], dtype=tf.float32)
                new_state = tf.convert_to_tensor([new_state], dtype=tf.float32)
                reward = tf.convert_to_tensor([reward], dtype=tf.float32)

                probs = self.actor_model.predict(cur_state)
                action_probs = tfp.distributions.Categorical(probs=probs)
                log_prob = action_probs.log_prob(action)

                state_value = self.critic_model.predict(cur_state)
                state_value_ = self.critic_model.predict(new_state)

                adv = reward + self.gamma * state_value_* (1-done) - state_value
                actor_loss = -log_prob*adv

                gradient = tape.gradient(actor_loss, self.actor_model.trainable_variables)
                self.actor_model.optimizer.apply_gradients(zip(
                    gradient, self.actor_model.trainable_variables))

    def _train_critic(self, samples):
        with tf.GradientTape(persistent=True) as tape:
            for sample in samples:
                cur_state, action, reward, new_state, done = sample

                cur_state = tf.convert_to_tensor([cur_state], dtype=tf.float32)
                new_state = tf.convert_to_tensor([new_state], dtype=tf.float32)
                reward = tf.convert_to_tensor([reward], dtype=tf.float32)

                state_value = self.critic_model.predict(cur_state)
                state_value_ = self.critic_model.predict(new_state)

                adv = reward + self.gamma * state_value_* (1-done) - state_value
                critic_loss = adv**2

                gradient = tape.gradient(critic_loss, self.critic_model.trainable_variables)
                self.critic_model.optimizer.apply_gradients(zip(
                    gradient, self.critic_model.trainable_variables))

    def train(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

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
        # escoger una acción del softmax y retornar la acción <self.action>
        # asociada a esa probabilidad
        # Verificar que la acción sea posible, sino escoger otra (?)


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

    def print_memory(self):
        for m in self.memory:
            cur_state, action, reward, new_state, done = m
            print(f'cur_state: {cur_state}')
            print(f'action: {action}')
            print(f'reward: {reward}')
            print(f'new_state: {new_state}')
            print(f'done: {done}')
            print()

    def save_memory(self):
        list_state = []
        list_action = []
        list_reward = []
        list_new_state = []
        list_done = []
        for m in self.memory:
            cur_state, action, reward, new_state, done = m
            list_state.append(cur_state)
            list_action.append(action)
            list_reward.append(reward)
            list_new_state.append(new_state)
            list_done.append(done)
        
        np.savez_compressed('data.npz', LS=list_state, LA=list_action,
                            LR=list_reward, LN=list_new_state, LD=list_done)

    def load_memory(self):
        data = np.load('data.npz', allow_pickle=True)
        
        list_state = data['LS']
        list_action = data['LA']
        list_reward = data['LR']
        list_new_state = data['LN']
        list_done = data['LD']

        print((list_reward[0]))
        print((list_done[0]))

        for i in range(len(list_state)):
            cur_state = list(map(list, list_state[i]))
            action    = tuple(list_action[i])
            reward    = list_reward[i]
            new_state = list(map(list, list_new_state[i]))
            done      = list_done[i]

            self.remember(cur_state, action, reward, new_state, done)
