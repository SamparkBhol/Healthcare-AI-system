# Healthcare AI System

## Overview

Welcome to my Healthcare AI System project! I developed this advanced AI system to enhance healthcare through autonomous agents, causal inference, Neural Architecture Search (NAS), and edge deployment. The system integrates state-of-the-art tools and frameworks to provide a comprehensive solution for training, evaluating, and deploying AI models in healthcare settings.

## Project Structure

Here's a quick rundown of the project's structure:

- **`agents/`**: This directory contains scripts for training and interacting with AI agents.
  - `agent_training.py`: Script to train healthcare agents using MuZero.
  - `agent_interaction.py`: Manages the logic for agent-patient interactions.
  - `agent_lifelong_learning.py`: Implements lifelong learning for agents to continuously improve.
  - `meta_learning.py`: Contains meta-learning scripts to enable faster adaptation.

- **`causal_inference/`**: These scripts handle causal inference and counterfactual simulations.
  - `causal_model.py`: Builds and reasons with causal models.
  - `counterfactual_simulation.py`: Simulates counterfactual healthcare scenarios to assess various treatments.
  - `inference_utils.py`: Provides utilities for integrating DoWhy and Pyro for causal modeling.

- **`NAS/`**: Focuses on Neural Architecture Search and AutoML.
  - `nas_train.py`: Responsible for NAS training and model generation.
  - `nas_model_search.py`: Uses AutoML for searching optimal NAS models.
  - `efficient_net_builder.py`: Builds and optimizes models using EfficientNet.
  - `ray_tune_optimization.py`: Performs hyperparameter tuning using Ray Tune.

- **`environment/`**: Contains the custom simulation environment and patient scenario generators.
  - `healthcare_env.py`: Implements a custom OpenAI Gym environment for healthcare simulations.
  - `env_config.yaml`: Configuration file for setting up the healthcare environment.
  - `patient_generator.py`: Generates patient scenarios and symptoms for agent interactions.

- **`edge_deployment/`**: Scripts for deploying models to edge devices.
  - `model_converter.py`: Converts models for deployment on edge devices, such as TensorFlow Lite or ONNX.
  - `edge_optimization.py`: Optimizes neural networks for mobile and IoT devices.
  - `edge_test.py`: Tests model performance on edge devices.

- **`utils/`**: Utility scripts for various tasks like data loading, evaluation, logging, and visualization.
  - `data_loader.py`: Provides utilities for loading data within the healthcare environment.
  - `evaluation.py`: Contains metrics and methods for evaluating agents and models.
  - `logger.py`: Configures logging for training and interaction processes.
  - `visualization.py`: Tools for visualizing agent decisions, models, and causal graphs.

- **`requirements.txt`**: Lists all the dependencies required for the project.

- **`main.py`**: The main entry point to run the entire system.

## Installation

To set up the project environment, simply install the required dependencies listed in `requirements.txt`:

pip install -r requirements.txt

## Usage

1. **Configuration**: Make sure to configure the `env_config.yaml` file in the `environment/` directory according to your healthcare environment needs.

2. **Running the Project**: Execute `main.py` to run the system. You can specify your configuration file as follows:

python main.py --config path_to_env_config.yaml

3. **Monitoring**: Logs will be stored in the `logs/` directory. You can review these logs to monitor the training and interaction processes.

4. **Evaluation and Visualization**: After running the system, you can evaluate the performance metrics and visualize the results using the `utils/visualization.py` script.

## Project Details

- **Agents**: My agents are trained using MuZero and designed to interact with a simulated healthcare environment. I've incorporated lifelong learning and meta-learning to continuously improve their performance.

- **Causal Inference**: The causal inference module I've developed builds causal models and simulates counterfactual scenarios to better understand the effects of different treatments.

- **Neural Architecture Search (NAS)**: The NAS component automates the process of searching for efficient neural network architectures and optimizes them for edge deployment.

- **Edge Deployment**: I've ensured that models are optimized and converted for deployment on edge devices, such as mobile phones and IoT devices, to ensure real-world applicability.

## Contributing

I welcome contributions to this project! Feel free to submit issues and pull requests on the GitHub repository. Please make sure to follow the coding standards and guidelines provided.

