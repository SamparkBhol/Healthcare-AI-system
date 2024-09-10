import tensorflow as tf
import onnx
import tf2onnx
import os

class ModelConverter:
    def __init__(self, model_path, output_dir="converted_models"):
        self.model_path = model_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def convert_to_tflite(self):
        # Load the TensorFlow model
        model = tf.keras.models.load_model(self.model_path)

        # Convert the model to TensorFlow Lite format
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        # Save the converted model
        tflite_model_path = os.path.join(self.output_dir, "model.tflite")
        with open(tflite_model_path, "wb") as f:
            f.write(tflite_model)

        print(f"TensorFlow Lite model saved to {tflite_model_path}")
        return tflite_model_path

    def convert_to_onnx(self):
        # Load the TensorFlow model
        model = tf.keras.models.load_model(self.model_path)

        # Convert the model to ONNX format
        onnx_model, _ = tf2onnx.convert.from_keras(model)

        # Save the ONNX model
        onnx_model_path = os.path.join(self.output_dir, "model.onnx")
        onnx.save_model(onnx_model, onnx_model_path)

        print(f"ONNX model saved to {onnx_model_path}")
        return onnx_model_path

if __name__ == "__main__":
    converter = ModelConverter(model_path="path_to_model.h5")
    converter.convert_to_tflite()
    converter.convert_to_onnx()
