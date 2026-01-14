#!/usr/bin/env python3
"""Test the phonemizer ONNX model."""

import onnxruntime as ort
import numpy as np
from pathlib import Path

# Load model
model_path = Path("models/TTS/phomenizer_en.onnx")
session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])

# Inspect inputs and outputs
print("Phonemizer Model Structure:")
print("\nInputs:")
for inp in session.get_inputs():
    print(f"  {inp.name}: shape={inp.shape}, type={inp.type}")

print("\nOutputs:")
for out in session.get_outputs():
    print(f"  {out.name}: shape={out.shape}, type={out.type}")

# Try to run it with sample text
print("\n\nTesting with 'hello':")

# Try different input formats
test_inputs = [
    ("Character codes", np.array([[ord(c) for c in "hello"]], dtype=np.int64)),
    ("Single string", np.array(["hello"])),
]

for name, input_data in test_inputs:
    try:
        print(f"\n{name}: {input_data.shape if hasattr(input_data, 'shape') else type(input_data)}")
        result = session.run(None, {session.get_inputs()[0].name: input_data})
        print(f"  Success! Output shape: {result[0].shape if hasattr(result[0], 'shape') else type(result[0])}")
        print(f"  Output: {result[0]}")
    except Exception as e:
        print(f"  Error: {e}")
