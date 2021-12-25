import layout as ly
import actor_critic as ac
MaxIter = 5000

def main(MaxIter=5000):
    S, H = (4,4)
    env = ly.Enviroment(S,H)
    actor_critic = ac.ActorCritic(env)

    cur_state = env.create_env(N=7)
    env.show_state(cur_state)

    for epoch in range(MaxIter):
        possible_actions = env.get_actions(cur_state)
        action = actor_critic.act(cur_state, possible_actions)
        bad_pos = env.get_bad_positions(cur_state)
        print(f'Iter: {epoch}')
        print(bad_pos)
        env.show_state(cur_state)

        new_state, reward, done = env.step(action)

        actor_critic.remember(cur_state, action, new_state, reward, done)

        actor_critic.train()

        cur_state = new_state

    actor_critic.metricas()


if __name__ == "__main__":
	main(MaxIter)