import threading
from game import FlappyBirdGame
from ai import DQNAgent, EPSILON, EPSILON_DECAY, EPSILON_MIN

NUM_AGENTS = 3
EPISODES = 1000

def train_agent(agent_id):
    agent = DQNAgent()
    game = FlappyBirdGame(agent_id)

    global EPSILON

    for episode in range(EPISODES):
        state = game.get_state()
        done = False

        while not done:
            game.render()  
            action = agent.select_action(state)
            next_state, reward, done = game.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.learn()
            state = next_state

        EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)

if __name__ == "__main__":
    threads = [threading.Thread(target=train_agent, args=(i,)) for i in range(NUM_AGENTS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
