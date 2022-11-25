from keras.models import Sequential
from keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import copy
import numpy as np
import pandas as pd
from data import generate_ann_state_lm


class ActorCritic:
    def __init__(self, env):
        self.env    = env
        self.gamma  = 1

        # ===================================================================== #
        #                               Actor Model                             #
        # Chain rule: find the gradient of chaging the actor network params in  #
        # getting closest to the final value network predictions, i.e. de/dA    #
        # Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
        # ===================================================================== #

        self.memory = []
        self.actor_file = f"model/actor_file_{self.env.get_shape()[0]}x{self.env.get_shape()[1]}_N_20.h5"
        self.actor_model = self.create_actor_model()
        self.train_policy_iterations = 3
        self.target_kl = 5
        self.actor_history_loss = []

        # ===================================================================== #
        #                              Critic Model                             #
        # ===================================================================== #

        self.critic_file = f'model/critic_file_{self.env.get_shape()[0]}x{self.env.get_shape()[1]}_N_20.h5'
        self.critic_model = self.create_critic_model()
        self.train_value_iterations = 1
        self.critic_history_loss = []

    # ========================================================================= #
    #                              Model Definitions                            #
    # ========================================================================= #

    def create_actor_model(self):
        try     : return tf.keras.models.load_model(self.actor_file)
        except  : 
            S, H = self.env.get_shape()
            n_actions = S * (H - 1)

            model = Sequential([
                Flatten(),
                Dense(128, activation='relu'),
                Dense(256, activation='relu'),
                Dense(128, activation='relu'),
                Dense(n_actions, activation='softmax')
            ])

            model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
            model.build(input_shape=(1, S, H))
            model.save(self.actor_file)
            return model

    def create_critic_model(self):
        try:
            return tf.keras.models.load_model(self.critic_file)
        except: 
            S, H = self.env.get_shape()

            model = Sequential([
                Flatten(),
                Dense(100, activation='relu'),
                Dense(150, activation='relu'),
                Dense(100, activation='relu'),
                Dense(1, activation='linear')
            ])
            
            model.compile(optimizer='adam',
                        loss='mse',
                        metrics=['mse'])
            model.build(input_shape=(1, S, H+2))
            model.save(self.critic_file)
            return model

    # ========================================================================= #
    #                               Model Training                              #
    # ========================================================================= #

    def transform_states_actor(self,states):
        rows, columns = self.env.get_shape()
        list_states = []
        for state in states:
            st = [generate_ann_state_lm(state, rows)]
            st = np.array(st)[:, :(rows)*columns]
            st = np.expand_dims(np.array(st), axis=2)
            st.shape = (st.shape[0], columns, rows)
            list_states.append(st)

        return np.array(list_states[:-1], dtype=np.float32)

    def index_of_actions(self,actions):
        rows, columns = self.env.get_shape()
        index_list = [(i, j) for i in range(rows)
                    for j in range(columns) if i != j]
        i_act = []
        for action in actions:
            i_act.append(index_list.index(action))
        return i_act

    def logprobabilities(self,logits, a):
        rows, columns = self.env.get_shape()
        num_actions = (rows-1)*columns
        logprobabilities_all = tf.nn.log_softmax(logits)
        logprobability = tf.reduce_sum(
            tf.one_hot(a, num_actions) * logprobabilities_all, axis=1
        )
        return logprobability

    def train_actor(self,states, actions, advantages, clip_ratio=0.9):
        size = len(states) - 1
        states_format = self.transform_states_actor(states)
        i_actions = self.index_of_actions(actions)

        old_log_p = advantages = tf.convert_to_tensor(advantages, dtype=np.float32)
        
        with tf.GradientTape() as tape:
            log_p = [None] * size
       
            for i in range(size):
                log_p[i] = (self.logprobabilities(self.actor_model(states_format[i]), i_actions[i]))

            ratio = tf.exp ( 
            log_p - old_log_p
            )
            min_advantage = tf.where(
            advantages < 0,
            (1 - clip_ratio) * advantages,
            (1 + clip_ratio) * advantages,
            )
            
            actor_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantages, min_advantage)
            )
        
            grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)    
        self.actor_model.optimizer.apply_gradients(zip(grads, self.actor_model.trainable_variables))
        self.actor_model.save(self.actor_file)

        kl = tf.reduce_mean(
            old_log_p - log_p
        )
        kl = tf.reduce_sum(kl)
        return actor_loss.numpy(), kl.numpy()

    def transform_states_critic(self, states):
        rows, columns = self.env.get_shape()
        list_states = []
        for state in states:
            st = [generate_ann_state_lm(state, rows)]
            st = np.array(st)[:, :(rows+2)*columns]
            st = np.expand_dims(np.array(st), axis=2)
            st.shape = (st.shape[0], columns, rows+2)
            list_states.append(st)
        return np.array(list_states[1:])

    def train_critic(self, states, rewards):
        states_format = self.transform_states_critic(states)

        with tf.GradientTape() as tape:
            # Valor absoluto de las recompensas: [4,3,2,1]
            # returns = tf.constant(list(map(abs, rewards)), dtype=np.float32)
            returns = tf.constant(rewards, dtype=np.float32)

            error = []
            for i, st in enumerate(states_format):
                error.append(returns[i] - self.critic_model(st))

            critic_loss = tf.reduce_mean(list(map(lambda x: x**2, error)))

            grads = tape.gradient(
                critic_loss, self.critic_model.trainable_variables)
            self.critic_model.optimizer.apply_gradients(
                zip(grads, self.critic_model.trainable_variables))
        return critic_loss.numpy()
 
    def fit(self,X,y,verbose=False,save_files=False):

        states     = X['states']
        advantages = X['advantages']
        actions    = y['actions']
        rewards    = y['rewards']

        # ac_loss, kl = self.train_actor(states, actions, advantages)
        # self.actor_history_loss.append(ac_loss)


        cr_loss = self.train_critic(states, rewards)
        self.critic_history_loss.append(cr_loss)

        if verbose:
            print('################################')
            print(f'Actor loss : {ac_loss}')
            print(f'Critic loss: {cr_loss}')
            print('################################')
        
        if save_files: self.save_graph()

        # # Se entrena actor
        # for i in range(X.shape[0]):
        #     print(f'    {i}     ')
        #     states     = X['states'].iloc[i]
        #     advantages = X['advantages'].iloc[i]
        #     actions    = y['actions'].iloc[i]
        #     rewards    = y['rewards'].iloc[i]

        #     for _ in range(self.train_policy_iterations):
        #         ac_loss, kl = self.train_actor(states, actions, advantages)
        #         if verbose and _ in [0, self.train_policy_iterations//2 ,self.train_policy_iterations-1]: 
        #             print(_, ac_loss, sep='\t')
        #         if kl > 1.5 * self.target_kl:
        #             # Early Stopping
        #             break
        #     self.actor_history_loss.append(ac_loss)
                
        #     # Se entrena el critico
        #     for _ in range(self.train_value_iterations):
        #         cr_loss = self.train_critic(states, rewards)
        #         if verbose and _ in [0, self.train_value_iterations//2 ,self.train_value_iterations-1]: 
        #             print(_, cr_loss, sep='\t')
        #     self.critic_history_loss.append(cr_loss)
        return
 
    # ========================================================================= #
    #                              Model Predictions                            #
    # ========================================================================= #

    def run_episode(self,initial_state, max_steps):
        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)


        initial_state_shape = initial_state.shape
        state = initial_state

        for t in tf.range(max_steps):
            # Convert state into a batched tensor (batch size = 1)
            state = tf.expand_dims(state, 0)

            # Run the model and to get action probabilities and critic value
            action_logits_t, value = self.critic_model(state)

            # Sample next action from the action probability distribution
            action = tf.random.categorical(action_logits_t, 1)
            action_probs_t = self.actor_model(action_logits_t)

            # Store critic values
            values = values.write(t, tf.squeeze(value))

            # Store log probability of the action chosen
            action_probs = action_probs.write(t, action_probs_t[0, action])

            # Apply action to the environment to get next state and reward
            state, reward, done = tf_env_step(action)
            state.set_shape(initial_state_shape)

            # Store reward
            rewards = rewards.write(t, reward)

            if tf.cast(done, tf.bool): break

        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()

        return action_probs, values, rewards

    def actor_predict(self,state):
        rows, columns = self.env.get_shape()
        st = [generate_ann_state_lm(state, rows)]
        st = np.array(st)[:, :(rows)*columns]
        st = np.expand_dims(np.array(st), axis=2)
        st.shape = (st.shape[0], columns, rows)

        vector = self.actor_model(st)
        index_max = []
        for i in range(len(st)):
            index_max.append(np.where(vector[i] == max(vector[i]))[0][0])
        return index_max

    def get_action(self,y_pred):
        rows,columns = self.env.get_shape()
        list_actions = [(i,j) for i in range(rows) for j in range(columns) if i!=j ] 
        actions = []
        for index in y_pred:
            actions.append(list_actions[index])
        return actions

    def critic_predict(self,state):
        rows, columns = self.env.get_shape()
        st = [generate_ann_state_lm(state, rows)]
        st = np.array(st)[:, :(rows+2)*columns]
        st = np.expand_dims(np.array(st), axis=2)
        st.shape = (st.shape[0], columns, rows+2)

        steps = self.critic_model(st)
        
        return np.array([int(s) for s in steps])
    
    def predict(self,X):
        data = {
            'actions' : [],
            'rewards' : []
        }
        for i in range(X.shape[0]):
            states     = X['states'].iloc[i]
            data['actions'].append( [ self.get_action(self.actor_predict(st))[0] for st in states] )
            data['rewards'].append( [ self.critic_predict(st)[0] for st in states] )
    
        return pd.DataFrame(data, columns=['actions','rewards'])   

    # ========================================================================= #
    #                                   Solve                                   #
    # ========================================================================= #
    def solve(self, Layout,solver,n_pasos=1):
        layout = Layout
        done = False;except_flag = False;count_actions=0;reward_acum=0
        current_state = layout.stacks

        states  = [current_state]
        rewards = []
        actions = []
        advantages = []
    
        while not done and count_actions<n_pasos:
            action = self.get_action(self.actor_predict(current_state))[0]           
            
            new_state, _, done = self.env.step(layout, action)
            actions.append(action)

            # Calculamos malas posiciones estado actual y siguiente
            BP_i  = len( self.env.get_bad_positions(current_state))
            BP_i_ = len( self.env.get_bad_positions(new_state))

            # Calculamos recompensa inmediata
            if done: reward = -len(solver(copy.deepcopy(layout))) - 1
            else   : reward = (BP_i - BP_i_) - 1
            
            #print(f'# {done}-> Rwds_acum: {reward_acum} - Rwds: {reward}')
            reward_acum += reward

            Vs  = self.critic_predict(current_state)
            Vs_ = self.critic_predict(new_state)    # Valor estado siguiente
            
            current_state  = new_state
            count_actions += 1
            states.append(copy.deepcopy(current_state))
            rewards.insert(0,reward_acum)
            advantages.append(reward + self.gamma * Vs_ - Vs)

        return states,actions,advantages,rewards,done

    # ========================================================================= #
    advanta#                                   Debug                                   #
    # ========================================================================= #
    def save_graph(self):
        rows,columns = self.env.get_shape()
        
        # Grafica Actor
        plt.figure(0)
        plt.title(f"Actor Model - {rows}x{columns}", size=16)
        plt.rcParams["figure.figsize"] = (20,15)

        plt.xlabel("Iter", size = 12,)
        plt.ylabel("Loss", size = 12)

        X_plot = range(len(self.actor_history_loss))
        y_plot = self.actor_history_loss
        plt.plot(X_plot, y_plot, label='Actor loss')

        # plt.vlines(x = [n for n in range(n_iter,N*n_iter,n_iter)],ymin = 0, ymax = max(y_plot), 
        #     colors = 'grey', 
        #     linestyles='--')

        plt.legend(loc='lower right')
        plt.savefig(f'img\\actor_loss_{rows}x{columns}.png')
        plt.close()
        
        # Grafica Critic
        plt.figure(1)
        plt.title(f"Critic Model - {rows}x{columns}", size=16)
        plt.rcParams["figure.figsize"] = (20,15)

        plt.xlabel("Iter", size = 12,)
        plt.ylabel("Loss", size = 12)

        X_plot = range(len(self.critic_history_loss))
        y_plot = self.critic_history_loss
        plt.plot(X_plot, y_plot, label='critic loss')

        plt.legend(loc='lower right')
        plt.savefig(f'img\\critic_loss_{rows}x{columns}.png')
        plt.close()

        np.savetxt('data\\critic_history_loss.csv',
            X=self.critic_history_loss,
            delimiter=','
            )
        np.savetxt('data\\actor_history_loss.csv',
            X=self.actor_history_loss,
            delimiter=','
            )
    

def main():
    print("hola mundo")


if __name__ == "__main__":
    main()
