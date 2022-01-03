import copy
import actor_critic as ac
from enviroment import Enviroment
from greedy import greedy_solve

MaxIter = 5000
GreedyIter = 2500

def main():
    S, H = (5,7)
    env = Enviroment(S,H)
    actor_critic = ac.ActorCritic(env)

    # Loop de entrenamiento mediante Greedy
    for _ in range(1):
        layout = env.create_env(N=20)
        layout_copy = copy.deepcopy(layout)

        actions = greedy_solve(layout) # Se consigue la lista de acciones que permiten resolver el layout

        initial_state = layout_copy.stacks
        solved_state = layout.stacks
        env.show_state(initial_state)
        print(f"actions: {actions}")
        env.show_state(solved_state)

        env.layout = layout_copy # Se vuelve al estado original del layout antes de ser resuelto

        for action in actions:
            cur_state = copy.deepcopy(env.layout.stacks)
            new_state, reward, done = env.step(action)

            print(f"cur_state:", cur_state)
            print("action:", action)
            print(f"new_state:", new_state)

            actor_critic.remember(cur_state, action, reward, new_state, done)
            actor_critic.train()

    layout = env.create_env(N=20)

    for epoch in range(MaxIter):
        possible_actions = env.get_actions(cur_state)
        action = actor_critic.act(cur_state, possible_actions)
        bad_pos = env.get_bad_positions(cur_state)
        print(f'Iter: {epoch}')
        print(bad_pos)
        env.show_state(cur_state)

        new_state, reward, done = env.step(action)  

        actor_critic.remember(cur_state, action, reward, new_state, done)

        actor_critic.train()

        cur_state = new_state

    actor_critic.metrics()

if __name__ == "__main__":
	main()