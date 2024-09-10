import dowhy
from dowhy import CausalModel
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

class InferenceUtils:
    @staticmethod
    def create_dowhy_causal_model(data, treatment, outcome, confounders):
        # Create a causal model using DoWhy
        causal_model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            common_causes=confounders
        )
        return causal_model

    @staticmethod
    def estimate_causal_effect(causal_model):
        # Estimate causal effect using DoWhy
        identified_estimand = causal_model.identify_effect()
        estimate = causal_model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
        return estimate

    @staticmethod
    def fit_pyro_model(model, data, guide=None, num_steps=1000):
        # Train a Pyro probabilistic model
        guide = guide or AutoDiagonalNormal(model)
        optim = Adam({"lr": 0.01})
        svi = SVI(model, guide, optim, loss=Trace_ELBO())

        pyro.clear_param_store()
        for step in range(num_steps):
            loss = svi.step(data)
            if step % 100 == 0:
                print(f"Step {step}: Loss = {loss}")

        return guide

    @staticmethod
    def perform_inference(model, guide, evidence, num_samples=1000):
        # Perform posterior inference using Pyro's Predictive
        predictive = pyro.infer.Predictive(model, guide=guide, num_samples=num_samples)
        posterior_samples = predictive(evidence)
        return posterior_samples

# Data utility functions
class DataUtils:
    @staticmethod
    def prepare_data(df, columns):
        # Preprocess data for causal inference (e.g., normalize, handle missing values)
        df = df[columns].dropna()
        return df

    @staticmethod
    def split_train_test(df, test_size=0.2):
        # Split data into train and test sets
        train_df = df.sample(frac=1 - test_size)
        test_df = df.drop(train_df.index)
        return train_df, test_df
