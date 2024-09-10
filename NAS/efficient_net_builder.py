import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

class EfficientNetBuilder:
    def __init__(self, input_shape=(224, 224, 3), num_classes=10, fine_tune_at=100):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.fine_tune_at = fine_tune_at

    def build(self):
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=self.input_shape)

        # Freeze the base model layers
        base_model.trainable = False

        # Create custom head for classification
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        return model

    def fine_tune(self, model):
        # Unfreeze layers for fine-tuning
        for layer in model.layers[0].layers[self.fine_tune_at:]:
            layer.trainable = True
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

class EfficientNetTrainer:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.model = None

    def train_model(self, fine_tune=False):
        efficient_net_builder = EfficientNetBuilder()
        self.model = efficient_net_builder.build()

        if fine_tune:
            self.model = efficient_net_builder.fine_tune(self.model)

        train_data, val_data = self.data_loader.load_data()

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model.fit(train_data, validation_data=val_data, epochs=10)

    def evaluate(self):
        _, test_data = self.data_loader.load_data(split='test')
        return self.model.evaluate(test_data)
