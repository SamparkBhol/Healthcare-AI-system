import tensorflow as tf
from tensorflow_model_optimization.sparsity import keras as sparsity
import onnx
import onnxruntime as ort
import os

class EdgeOptimizer:
    def __init__(self, model_path, output_dir="optimized_models"):
        self.model_path = model_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def apply_quantization(self):
        # Load the model
        model = tf.keras.models.load_model(self.model_path)

        # Convert the model to TensorFlow Lite with quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quantized_model = converter.convert()

        # Save the quantized model
        quantized_model_path = os.path.join(self.output_dir, "quantized_model.tflite")
        with open(quantized_model_path, "wb") as f:
            f.write(quantized_model)

        print(f"Quantized model saved to {quantized_model_path}")
        return quantized_model_path

    def apply_pruning(self, epochs=10):
        # Load the model
        model = tf.keras.models.load_model(self.model_path)

        # Apply pruning
        pruning_schedule = sparsity.PolynomialDecay(initial_sparsity=0.0, final_sparsity=0.5, begin_step=0, end_step=epochs * 1000)
        pruned_model = sparsity.prune_low_magnitude(model, pruning_schedule=pruning_schedule)

        # Save the pruned model
        pruned_model_path = os.path.join(self.output_dir, "pruned_model.h5")
        pruned_model.save(pruned_model_path)

        print(f"Pruned model saved to {pruned_model_path}")
        return pruned_model_path

    def optimize_onnx(self):
        # Load the ONNX model
        model_path = os.path.join(self.output_dir, "model.onnx")
        onnx_model = onnx.load(model_path)

        # Optimize using ONNX runtime
        sess_options = ort.SessionOptions()
        sess_options.optimized_model_filepath = os.path.join(self.output_dir, "optimized_model.onnx")

        # Run optimization
        ort.InferenceSession(onnx_model.SerializeToString(), sess_options)

        print(f"ONNX optimized model saved to {sess_options.optimized_model_filepath}")
        return sess_options.optimized_model_filepath

if __name__ == "__main__":
    optimizer = EdgeOptimizer(model_path="path_to_model.h5")
    optimizer.apply_quantization()
    optimizer.apply_pruning()
    optimizer.optimize_onnx()
