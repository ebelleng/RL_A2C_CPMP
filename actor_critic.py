import numpy as np

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

        # ===================================================================== #
        #                              Critic Model                             #
        # ===================================================================== #

    # ========================================================================= #
    #                              Model Definitions                            #
    # ========================================================================= #

    def create_actor_model(self):
        return

    def create_critic_model(self):
        return

    # ========================================================================= #
    #                               Model Training                              #
    # ========================================================================= #

    def remember(self, cur_state, action, reward, new_state, done):
        self.memory.append([cur_state, action, reward, new_state, done])

    def _train_actor(self):
        return

    def _train_critic(self):
        return

    def train(self):
        return

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

    def metrics():
        return 
        
    # ========================================================================= #
    #                                Debug                                      #
    # ========================================================================= #
    
    def show(self):
        print("Hello world")
