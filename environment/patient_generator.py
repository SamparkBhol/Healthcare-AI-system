import numpy as np
import yaml

class PatientGenerator:
    def __init__(self, config):
        self.initial_state_range = config['initial_state_range']
        self.symptom_range = config['symptom_range']
        self.state_change_noise = config['state_change_noise']
        self.treatment_success_chance = config['treatment_success_chance']

    def generate_patient(self):
        # Generate a new patient with random initial state (e.g., age, heart rate, blood pressure, symptoms)
        patient = np.random.uniform(
            low=self.initial_state_range['low'],
            high=self.initial_state_range['high']
        )
        symptoms = np.random.uniform(
            low=self.symptom_range['low'],
            high=self.symptom_range['high']
        )
        return np.concatenate([patient, symptoms])

    def generate_state(self, current_state=None):
        if current_state is None:
            current_state = self.generate_patient()
        
        # Apply random state change to simulate progression of the patient's condition
        noise = np.random.uniform(
            low=-self.state_change_noise,
            high=self.state_change_noise,
            size=current_state.shape
        )
        new_state = current_state + noise
        new_state = np.clip(new_state, self.initial_state_range['low'], self.initial_state_range['high'])
        
        return new_state

    def apply_treatment(self, patient_state, treatment_type):
        # Apply a treatment to the patient and simulate the effect
        if np.random.rand() < self.treatment_success_chance:
            # Treatment is successful, patient's condition improves
            patient_state[-1] = 0  # Example: fever is gone
        else:
            # Treatment fails, condition remains the same or worsens
            noise = np.random.uniform(
                low=-self.state_change_noise,
                high=self.state_change_noise,
                size=patient_state.shape
            )
            patient_state += noise
            patient_state = np.clip(patient_state, self.initial_state_range['low'], self.initial_state_range['high'])
        
        return patient_state

    def simulate_episode(self):
        # Simulate a full episode for a single patient, with multiple treatments
        patient_state = self.generate_patient()
        for step in range(100):  # Max 100 steps in an episode
            # Randomly apply a treatment or observation
            treatment_type = np.random.choice(['A', 'B', 'observe'])
            if treatment_type == 'observe':
                patient_state = self.generate_state(patient_state)
            else:
                patient_state = self.apply_treatment(patient_state, treatment_type)

            print(f"Step {step}: Patient state: {patient_state}")

        return patient_state

# Generalized patient generation process
if __name__ == "__main__":
    with open('env_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    patient_generator = PatientGenerator(config['patient_generator'])
    patient_generator.simulate_episode()
