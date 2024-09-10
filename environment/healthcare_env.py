import gym
from gym import spaces
import numpy as np
import yaml
from patient_generator import PatientGenerator

class HealthcareEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config_path="env_config.yaml"):
        super(HealthcareEnv, self).__init__()

        # Load environment configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.patient_generator = PatientGenerator(self.config['patient_generator'])
        
        # Define the action and observation space
        self.action_space = spaces.Discrete(self.config['action_space'])
        self.observation_space = spaces.Box(
            low=np.array(self.config['observation_space']['low']),
            high=np.array(self.config['observation_space']['high']),
            dtype=np.float32
        )
        
        # Initialize state
        self.current_patient = None
        self.state = None
        self.reset()

    def step(self, action):
        # Apply the action to the environment and get feedback
        reward, done = self._take_action(action)
        self.state = self._get_next_state()
        info = {}

        return np.array(self.state), reward, done, info

    def _take_action(self, action):
        # Define how the action affects the state and reward
        if action == 0:
            # Apply treatment A
            return self._apply_treatment('A')
        elif action == 1:
            # Apply treatment B
            return self._apply_treatment('B')
        elif action == 2:
            # Apply no treatment (observation)
            return self._apply_observation()
        else:
            return 0, False

    def _apply_treatment(self, treatment_type):
        # Placeholder for logic to apply treatment
        # Modify the patient’s state and return reward
        success = np.random.rand() < 0.7  # 70% chance of treatment success
        if success:
            reward = 1
            done = True
        else:
            reward = -1
            done = False

        return reward, done

    def _apply_observation(self):
        # Logic for observation with no immediate treatment
        reward = 0
        done = False
        return reward, done

    def _get_next_state(self):
        # Placeholder for updating the environment state
        # Simulate patient health metrics after treatment
        return self.patient_generator.generate_state()

    def reset(self):
        # Reset the environment and generate a new patient
        self.current_patient = self.patient_generator.generate_patient()
        self.state = self._get_next_state()
        return np.array(self.state)

    def render(self, mode='human'):
        # Display the current patient state
        if mode == 'human':
            print(f"Patient state: {self.state}")

    def close(self):
        # Clean up the environment
        pass

if __name__ == "__main__":
    env = HealthcareEnv()
    env.reset()
    env.render()
