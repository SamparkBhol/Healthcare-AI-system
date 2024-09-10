from tensorflow.keras import layers, models
import tensorflow as tf

class NASModelSearch:
    def __init__(self, search_space):
        self.search_space = search_space

    def build_model(self, conv_layers, filter_size, kernel_size, activation):
        model = models.Sequential()
        model.add(layers.Input(shape=(224, 224, 3)))  # Example input shape
        
        # Add convolutional layers based on search parameters
        for _ in range(conv_layers):
            model.add(layers.Conv2D(filters=filter_size, kernel_size=kernel_size, activation=activation))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        
        # Add dense layers
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        return model

    def search(self, hyperparams=None):
        # Perform a grid search over the provided hyperparameter space
        if hyperparams is None:
            hyperparams = {
                "conv_layers": 2,
                "filter_size": 64,
                "kernel_size": 3,
                "activation": "relu"
            }
        
        # Build the model with the given hyperparameters
        model = self.build_model(
            conv_layers=hyperparams['conv_layers'],
            filter_size=hyperparams['filter_size'],
            kernel_size=hyperparams['kernel_size'],
            activation=hyperparams['activation']
        )
        
        return model

    def evaluate_model(self, model, data):
        # Evaluate the model on validation data
        val_loss, val_acc = model.evaluate(data['val'])
        return val_acc

    def search_best_model(self, search_space, data_loader):
        best_model = None
        best_acc = 0.0

        # Grid search over search space
        for conv_layers in search_space['conv_layers']:
            for filter_size in search_space['filter_size']:
                for kernel_size in search_space['kernel_size']:
                    for activation in search_space['activation']:
                        # Build and compile model
                        model = self.build_model(conv_layers, filter_size, kernel_size, activation)
                        self.compile_model(model)

                        # Train and evaluate the model
                        train_data, val_data = data_loader.load_data()
                        model.fit(train_data, epochs=10)
                        acc = self.evaluate_model(model, {'val': val_data})

                        if acc > best_acc:
                            best_acc = acc
                            best_model = model

        return best_model

