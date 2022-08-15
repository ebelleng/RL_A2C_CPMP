import numpy as np
import random
import copy

from keras.models import Model
from keras.layers import Input, Dense
from keras.layers.merge import Add
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
import tensorflow_probability as tfp

from utils import prepare
from permutational_layer import *
from data import *

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
        _, self.target_actor_model, _ = self.create_actor_model()

        # ===================================================================== #
        #                              Critic Model                             #
        # ===================================================================== #
        
        self.critic_model = self.create_critic_model()
        self.train_critic_model()
        
    # ========================================================================= #
    #                              Model Definitions                            #
    # ========================================================================= #

    def create_actor_model(self):
        n_actions = self.env.get_shape()[0] *  (self.env.get_shape()[0] - 1)
        S, H = self.env.get_shape()

        state_input = Input(shape=S * H)
        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        output = Dense(n_actions, activation='softmax')(h3) #action
        # [(0,1), (0,2), (1,0), (1,2) ]

        model = Model(state_input, output)
        adam  = Adam(learning_rate=0.001)
        model.compile(optimizer=adam)

        return state_input, model, output

    def create_critic_model(self):
        rows, columns = self.env.get_shape()

        perm_layer1 = PermutationalLayer(PermutationalEncoder(PairwiseModel((rows+2,), repeat_layers(Dense, [32, 32], activation='relu')), columns), name='permutational_layer1')
        perm_layer2 = PermutationalLayer(PermutationalEncoder(PairwiseModel((32,), repeat_layers(Dense, [32, 32], activation='relu')), columns), name='permutational_layer2')
        perm_layer3 = PermutationalLayer(PermutationalEncoder(PairwiseModel((32,), repeat_layers(Dense, [32, 32], activation='relu')), columns), pooling=maximum, name='permutational_layer3')
        
        dense1 = Dense(128, activation="relu", name="layer1")
        dense2 = Dense(64, activation="relu", name="layer2")
        dense3 = Dense(1, activation="linear", name="layer3")

        inputs = []
        for i in range(columns):
            inputs.append(Input((rows+2,), name='stack_'+str(i)))

        outputs = perm_layer1(inputs)
        outputs = perm_layer2(outputs)
        outputs = perm_layer3(outputs)
        outputs = dense1(outputs)
        outputs = dense2(outputs)
        output = dense3(outputs)

        model = Model(inputs, output)

        adam  = Adam(learning_rate=0.001)
        model.compile(loss='mse', optimizer=adam, metrics=['mse'])

        return model

    # ========================================================================= #
    #                               Model Training                              #
    # ========================================================================= #

    def remember(self, cur_state, action, reward, new_state, done):
        if cur_state is None or action is None or reward is None or new_state is None or done is None:
            print("HAY UN NONEEEEEEEE")
        self.memory.append([cur_state, action, reward, new_state, done])

    def _train_actor(self, samples, H): # policy_train
        for sample in samples:
            cur_state, action, reward, new_state, done = sample

            with tf.GradientTape(persistent=True) as tape:
                cur_state = prepare(cur_state, H)
                new_state = prepare(new_state, H)

                cur_state = tf.convert_to_tensor([cur_state], dtype=tf.float32)
                new_state = tf.convert_to_tensor([new_state], dtype=tf.float32)
                reward = tf.convert_to_tensor([reward], dtype=tf.float32)

                tf.debugging.check_numerics(cur_state, 'Checking cur_state train_actor')
                tf.debugging.check_numerics(new_state, 'Checking new_state train_actor')

                probs = self.actor_model(cur_state)
                action_probs = tfp.distributions.Categorical(probs=probs)
                log_prob = action_probs.log_prob(action)

                state_value = self.critic_model(cur_state)
                state_value_ = self.critic_model(new_state)

                adv = reward + self.gamma * state_value_* (1-done) - state_value
                actor_loss = -log_prob*adv

            gradient = tape.gradient(actor_loss, self.actor_model.trainable_variables)
            self.actor_model.optimizer.apply_gradients(zip(
                gradient, self.actor_model.trainable_variables))

    def _train_critic(self, samples, H):
        cont = 0
        archivo = open("train_critic.txt", 'w')
        for i, sample in enumerate(samples):
            cur_state, _, reward, new_state, done = sample
            print(f"Iter {cont} - {done}", end=' ')
            archivo.write(f"Iter {cont} - {done}")

            with tf.GradientTape(persistent=True) as tape:
                cur_state = prepare(cur_state, H)
                new_state = prepare(new_state, H)

                cur_state = tf.convert_to_tensor([cur_state], dtype=tf.float32)
                new_state = tf.convert_to_tensor([new_state], dtype=tf.float32)
                reward = tf.convert_to_tensor([reward], dtype=tf.float32)

                tf.debugging.check_numerics(cur_state, 'Checking cur_state train_critic')
                tf.debugging.check_numerics(new_state, 'Checking new_state train_critic')

                state_value = self.critic_model(cur_state)
                state_value_ = self.critic_model(new_state)
                

                adv = reward + self.gamma * state_value_* (1-done) - state_value
                critic_loss = adv**2
                print(f'{state_value}\t{state_value_}\t{adv}\t{critic_loss}')
                archivo.write(f'{state_value}\t{state_value_}\t{adv}\t{critic_loss}')

        
            gradient = tape.gradient(critic_loss, self.critic_model.trainable_variables)
            self.critic_model.optimizer.apply_gradients(zip(
                gradient, self.critic_model.trainable_variables))
            cont += 1
        archivo.close()

    def train(self, H, all = False):
        if not all:
            batch_size = 32
            if len(self.memory) < batch_size:
                return
        else:
            batch_size = len(self.memory)

        samples = random.sample(self.memory, batch_size)
        #samples = self.get_samples() 
        self._train_critic(samples, H)
        self._train_actor(samples, H)

    def train_critic_model(self):
        self.data_train = generate_data_greedy(51200)
        self.S, self.H = self.env.get_shape()
        for i in range(200):
            X,y = self.data_train
            Xtrain_input_array = InputEncoder(X, self.S, self.H)
            history=self.critic_model.fit(Xtrain_input_array, y, epochs=10, batch_size=1024, verbose=False)
            print(history.history['loss'][0])
        
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
        cur_state_copy = copy.deepcopy(cur_state)
        cur_state_copy = prepare(cur_state_copy, self.env.H)
        cur_state_copy = tf.convert_to_tensor([cur_state_copy], dtype=tf.float32)


        tf.print(cur_state_copy)
        tf.debugging.check_numerics(cur_state_copy, 'Checking cur_state_copy before probs')

        print(cur_state_copy.numpy())
        probs = self.actor_model(cur_state_copy)

        tf.debugging.check_numerics(probs, 'Checking probs')
        
        action_probabilities = tfp.distributions.Categorical(probs=probs)
        action = action_probabilities.sample()

        try: 
            x = tf.debugging.check_numerics(probs, 'Checking action try')
            return possible_actions[action.numpy()[0]]
        except:
            #print(action)
            x = tf.debugging.check_numerics(probs, 'Checking action except')
            return possible_actions[0]
    
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

    def get_missing_actions(i, samples):
        return

    def get_samples(self):
        mini_sample = []
        count = 0
        for sample in self.memory:
            cur_state, action, reward, new_state, done = sample
            mini_sample.append(sample)
            if done == 1:
                count += 1

            if done == 1 and count == 5:
                print("Cantidad de weas : ", len(mini_sample), count)
                return mini_sample

