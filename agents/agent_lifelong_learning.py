class LifelongLearningAgent:
    def __init__(self, base_agent):
        self.base_agent = base_agent
        self.experience_replay = []

    def store_experience(self, state, action, reward, next_state):
        # Store experience in a replay buffer for future learning
        self.experience_replay.append((state, action, reward, next_state))
        if len(self.experience_replay) > 10000:  # Limiting buffer size
            self.experience_replay.pop(0)

    def update_agent(self, params):
        # Update the agent's model using stored experiences in the buffer
        for state, action, reward, next_state in self.experience_replay:
            target = reward + 0.99 * max(self.base_agent.model(params, next_state))
            params = self.base_agent.train(state, target)
        return params

    def lifelong_learn(self, env, params, num_episodes=1000):
        for episode in range(num_episodes):
            state = env.reset()
            done = False

            while not done:
                action = self.base_agent.select_action(params, state)
                next_state, reward, done, _ = env.step(action)
                self.store_experience(state, action, reward, next_state)
                params = self.update_agent(params)
                state = next_state

        return params

# Mock usage with a basic agent
class BasicAgent:
    def __init__(self):
        pass

    def model(self, params, state):
        return state.sum()

    def train(self, state, target):
        # Dummy train function
        return {}

class MockEnv:
    def reset(self):
        return [0] * 5
    
    def step(self, action):
        return [0] * 5, 1, True, None
