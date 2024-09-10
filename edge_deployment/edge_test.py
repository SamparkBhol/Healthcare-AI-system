import tensorflow as tf
import onnxruntime as ort
import numpy as np

class EdgeTester:
    def __init__(self, model_path):
        self.model_path = model_path

    def test_tflite_model(self, input_data):
        # Load TensorFlow Lite model
        interpreter = tf.lite.Interpreter(model_path=self.model_path)
        interpreter.allocate_tensors()

        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Set input data
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Get the output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        return output_data

    def test_onnx_model(self, input_data):
        # Load ONNX model
        session = ort.InferenceSession(self.model_path)

        # Get model inputs and outputs
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # Run inference
        results = session.run([output_name], {input_name: input_data})
        return results[0]

if __name__ == "__main__":
    # Test TensorFlow Lite model
    tflite_tester = EdgeTester(model_path="converted_models/model.tflite")
    input_data = np.random.random_sample((1, 224, 224, 3)).astype(np.float32)
    tflite_result = tflite_tester.test_tflite_model(input_data)
    print(f"TensorFlow Lite model inference result: {tflite_result}")

    # Test ONNX model
    onnx_tester = EdgeTester(model_path="converted_models/model.onnx")
    onnx_result = onnx_tester.test_onnx_model(input_data)
    print(f"ONNX model inference result: {onnx_result}")
