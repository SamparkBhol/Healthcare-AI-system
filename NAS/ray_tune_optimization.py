import ray
from ray import tune
from nas_model_search import NASModelSearch
import tensorflow as tf

class RayTuneOptimizer:
    def __init__(self, search_space):
        self.search_space = search_space

    def objective(self, config):
        # Objective function for tuning, training, and evaluating the model
        nas_model_search = NASModelSearch(self.search_space)

        model = nas_model_search.search(config)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config["lr"]),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Load training and validation data
        data_loader = DataLoader()
        train_data, val_data = data_loader.load_data()

        model.fit(train_data, epochs=10, verbose=0)
        val_loss, val_acc = model.evaluate(val_data, verbose=0)
        tune.report(mean_accuracy=val_acc)

    def optimize(self, num_samples=10, max_concurrent_trials=4):
        search_space = {
            "conv_layers": tune.choice([1, 2, 3]),
            "filter_size": tune.choice([32, 64, 128]),
            "kernel_size": tune.choice([3, 5, 7]),
            "activation": tune.choice(["relu", "swish"]),
            "lr": tune.loguniform(1e-4, 1e-2)
        }

        # Run Ray Tune optimization
        analysis = tune.run(
            self.objective,
            config=search_space,
            num_samples=num_samples,
            resources_per_trial={"cpu": 2, "gpu": 1},
            max_concurrent_trials=max_concurrent_trials
        )

        best_config = analysis.get_best_config(metric="mean_accuracy", mode="max")
        return best_config
