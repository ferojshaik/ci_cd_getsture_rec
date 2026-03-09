"""
STEP 4: TFLite export

What this script does:
- Loads the trained SavedModel from saved_model/
- Converts it to TensorFlow Lite (.tflite) for the phone
- Optionally creates a quantized (int8) version for smaller size and faster inference
- Verifies the TFLite model by running it on a few test samples

Output:
- gesture_model.tflite       (float32, baseline)
- gesture_model_quant.tflite (int8, smaller/faster - use this on phone)
"""

import numpy as np
from pathlib import Path

import tensorflow as tf

from dataset import get_testing_dataset, NUM_FEATURES, index_to_label


# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
SAVED_MODEL_DIR = PROJECT_ROOT / "saved_model"
TFLITE_FLOAT_PATH = PROJECT_ROOT / "gesture_model.tflite"
TFLITE_QUANT_PATH = PROJECT_ROOT / "gesture_model_quant.tflite"


def convert_to_tflite(quantize=False):
    """
    Load SavedModel and convert to TFLite.
    quantize=True: int8 quantization (smaller, faster, ~same accuracy for this model).
    """
    # Convert from SavedModel directory (saved by trainer.py via model.export())
    converter = tf.lite.TFLiteConverter.from_saved_model(str(SAVED_MODEL_DIR))

    if quantize:
        # int8 quantization: weights and activations use 8-bit integers.
        # Representative dataset lets the converter estimate activation ranges.
        from dataset import get_training_dataset
        X_train, _, _ = get_training_dataset()
        def representative_dataset():
            for i in range(min(200, len(X_train))):
                yield [X_train[i : i + 1].astype(np.float32)]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        out_path = TFLITE_QUANT_PATH
    else:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        tflite_model = converter.convert()
        out_path = TFLITE_FLOAT_PATH

    out_path.write_bytes(tflite_model)
    print(f"Saved: {out_path}  ({len(tflite_model):,} bytes)")
    return out_path, tflite_model


def run_tflite_inference(tflite_path, X_sample):
    """Run TFLite model on one or more samples. Returns class indices (0, 1, or 2)."""
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Check if quantized (int8) or float
    input_dtype = input_details[0]["dtype"]
    output_dtype = output_details[0]["dtype"]
    is_quant = input_dtype in (np.int8, np.uint8)

    if X_sample.ndim == 1:
        X_sample = X_sample.reshape(1, -1)
    interpreter.resize_tensor_input(input_details[0]["index"], X_sample.shape)
    interpreter.allocate_tensors()

    if is_quant:
        # Quantized model: scale/zeropoint from input details
        scale, zero = input_details[0]["quantization"]
        X_int8 = (X_sample / scale + zero).astype(np.int8).clip(-128, 127)
        interpreter.set_tensor(input_details[0]["index"], X_int8)
    else:
        interpreter.set_tensor(input_details[0]["index"], X_sample.astype(np.float32))

    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]["index"])

    if is_quant:
        scale_out, zero_out = output_details[0]["quantization"]
        out = (out.astype(np.float32) - zero_out) * scale_out
    pred = np.argmax(out, axis=1)
    return pred, out


def verify_tflite(tflite_path, num_samples=5):
    """Run TFLite on a few test samples and print predicted vs true label."""
    X_test, y_test, _ = get_testing_dataset()
    if len(X_test) == 0:
        print("No test data to verify.")
        return
    n = min(num_samples, len(X_test))
    pred, probs = run_tflite_inference(tflite_path, X_test[:n])
    print(f"\nVerification ({tflite_path.name}), first {n} test samples:")
    for i in range(n):
        true_label = index_to_label(int(y_test[i]))
        pred_label = index_to_label(int(pred[i]))
        ok = "[OK]" if pred[i] == y_test[i] else "[X]"
        print(f"  true={true_label}, predicted={pred_label} {ok}")


def main():
    print("Step 4: TFLite export\n")

    if not SAVED_MODEL_DIR.is_dir():
        print(f"Error: SavedModel not found at {SAVED_MODEL_DIR}")
        print("Run trainer.py first to train and save the model.")
        return

    # 1. Float32 TFLite (baseline)
    print("Converting to TFLite (float32)...")
    convert_to_tflite(quantize=False)
    verify_tflite(TFLITE_FLOAT_PATH, num_samples=5)

    # 2. Quantized int8 TFLite (for phone)
    print("\nConverting to TFLite (int8 quantized)...")
    convert_to_tflite(quantize=True)
    verify_tflite(TFLITE_QUANT_PATH, num_samples=5)

    print("\nDone. Use gesture_model_quant.tflite on the phone for smaller size and faster inference.")
    print("On the phone: stream accelerometer -> build 39 features (same window/FFT) -> run this model.")


if __name__ == "__main__":
    main()
