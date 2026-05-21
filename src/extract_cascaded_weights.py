"""
===============================================================================
Cascaded Boost Controller — Weight Extraction Script (Version 1.0.0)
Author: Seyed Reza Rasaei

Description:
    Extracts trained neural network weights and normalization parameters
    and generates a C header file for deployment on STM32G431RB.

Features:
    - Supports Dense-only architectures (no BatchNorm)
    - Reads StandardScaler parameters from .npz file
    - Performs NumPy forward-pass verification
    - Generates a deterministic C inference function
    - Produces cascaded_model_weights.h
===============================================================================
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

MODEL_PATH = "cascaded_boost_model.h5"
SCALER_PATH = "scaler_params_cascaded.npz"
OUTPUT_FILE = "cascaded_model_weights.h"

# =============================================================================
# 1. Load Model and Scalers
# =============================================================================
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("Loading normalization parameters...")
sc = np.load(SCALER_PATH)
mean_X = sc["mean_X"]
scale_X = sc["scale_X"]
min_y = sc["min_y"]
max_y = sc["max_y"]
feat_min = sc["feat_min"]
feat_max = sc["feat_max"]

print(f"mean_X: {mean_X.round(4)}")
print(f"scale_X: {scale_X.round(4)}")
print(f"D1 range: [{feat_min[0]}, {feat_max[0]}]")
print(f"D2 range: [{feat_min[1]}, {feat_max[1]}]")

# =============================================================================
# 2. Identify Layers
# =============================================================================
print("\nModel layers:")
for layer in model.layers:
    print(f"  [{type(layer).__name__:<20}] {layer.name}")

shared_dense = [
    l for l in model.layers
    if isinstance(l, keras.layers.Dense) and l.name.startswith("Dense")
]

output_dense = [
    l for l in model.layers
    if isinstance(l, keras.layers.Dense) and l.name.startswith("Output")
]

all_dense = shared_dense + output_dense

print(f"\nShared Dense Layers: {[l.name for l in shared_dense]}")
print(f"Output Layers: {[l.name for l in output_dense]}")

# =============================================================================
# 3. NumPy Forward-Pass Verification
# =============================================================================
print("\nVerifying weight correctness...")

def numpy_forward_pass(X_raw, model, mean_X, scale_X, feat_min, feat_max):
    """Replicates the exact C inference pipeline using NumPy."""
    buf = (X_raw - mean_X) / scale_X

    shared = [
        l for l in model.layers
        if isinstance(l, keras.layers.Dense) and l.name.startswith("Dense")
    ]
    for layer in shared:
        W, b = layer.get_weights()
        buf = np.maximum(0.0, buf @ W + b)

    out_layers = [
        l for l in model.layers
        if isinstance(l, keras.layers.Dense) and l.name.startswith("Output")
    ]

    results = []
    for i, layer in enumerate(out_layers):
        W, b = layer.get_weights()
        raw = 1.0 / (1.0 + np.exp(-(buf @ W + b)))
        duty = raw * (feat_max[i] - feat_min[i]) + feat_min[i]
        duty = np.clip(duty, 0.05, 0.92)
        results.append(duty.flatten())

    return results

np.random.seed(0)
X_test_raw = np.column_stack([
    np.random.uniform(10, 30, 50),   # Vin
    np.random.uniform(30, 120, 50),  # Vint
    np.random.uniform(0, 5, 50),     # Iint
    np.random.uniform(40, 180, 50),  # Vout
    np.random.uniform(0, 5, 50),     # Iout
    np.random.uniform(48, 180, 50),  # Vref
    np.random.uniform(-30, 30, 50),  # error_Vout
])

X_test_norm = (X_test_raw - mean_X) / scale_X
pred_keras = model.predict(X_test_norm, verbose=0)
pred_d1_k = pred_keras[0].flatten()
pred_d2_k = pred_keras[1].flatten()

pred_np = numpy_forward_pass(X_test_raw, model, mean_X, scale_X, feat_min, feat_max)
pred_d1_np = pred_np[0]
pred_d2_np = pred_np[1]

pred_d1_k_duty = pred_d1_k * (feat_max[0] - feat_min[0]) + feat_min[0]
pred_d2_k_duty = pred_d2_k * (feat_max[1] - feat_min[1]) + feat_min[1]

diff_d1 = np.max(np.abs(pred_d1_k_duty - pred_d1_np))
diff_d2 = np.max(np.abs(pred_d2_k_duty - pred_d2_np))

print(f"Max D1 error: {diff_d1:.2e}")
print(f"Max D2 error: {diff_d2:.2e}")

if diff_d1 < 1e-5 and diff_d2 < 1e-5:
    print("Verification passed.")
else:
    print("WARNING: Verification failed.")

# =============================================================================
# 4. Generate C Header File
# =============================================================================
print(f"\nGenerating {OUTPUT_FILE}...")

max_hidden = max(l.get_weights()[0].shape[1] for l in shared_dense)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    f.write("/*\n")
    f.write(" * ======================================================================\n")
    f.write(" * cascaded_model_weights.h\n")
    f.write(" * Cascaded Boost Controller Weights — Version 1.0.0\n")
    f.write(" * Generated by extract_cascaded_weights.py\n")
    f.write(" *\n")

    arch_str = "Input(7)"
    for l in shared_dense:
        arch_str += f" -> Dense({l.get_weights()[0].shape[1]},ReLU)"
    arch_str += " -> [D1(sigmoid), D2(sigmoid)]"
    f.write(f" * Architecture: {arch_str}\n")

    total_params = sum(np.prod(w.shape) for l in all_dense for w in l.get_weights())
    f.write(f" * Parameters: {total_params:,}\n")
    f.write(f" * Flash usage: {total_params * 4 / 1024:.1f} KB\n")
    f.write(" * ======================================================================\n")
    f.write(" */\n\n")

    f.write("#ifndef CASCADED_MODEL_WEIGHTS_H\n")
    f.write("#define CASCADED_MODEL_WEIGHTS_H\n\n")
    f.write("#include <stdint.h>\n")
    f.write("#include <math.h>\n\n")

    f.write("#define NN_INPUT_SIZE 7U\n")
    f.write("#define NN_OUTPUT_SIZE 2U\n")
    f.write(f"#define NN_MAX_HIDDEN {max_hidden}U\n")
    f.write("#define D_MIN 0.05f\n")
    f.write("#define D_MAX 0.92f\n\n")

    feat_names = ["VIN", "VINT", "IINT", "VOUT", "IOUT", "VREF", "ERR_VOUT"]

    f.write("static const float NN_MEAN[NN_INPUT_SIZE] = {\n    ")
    f.write(",\n    ".join([f"{v:.8f}f /* {n} */" for v, n in zip(mean_X, feat_names)]))
    f.write("\n};\n\n")

    f.write("static const float NN_STD[NN_INPUT_SIZE] = {\n    ")
    f.write(",\n    ".join([f"{v:.8f}f /* {n} */" for v, n in zip(scale_X, feat_names)]))
    f.write("\n};\n\n")

    f.write(f"#define NN_D1_FEAT_MIN {feat_min[0]:.8f}f\n")
    f.write(f"#define NN_D1_FEAT_MAX {feat_max[0]:.8f}f\n")
    f.write(f"#define NN_D2_FEAT_MIN {feat_min[1]:.8f}f\n")
    f.write(f"#define NN_D2_FEAT_MAX {feat_max[1]:.8f}f\n\n")

    for layer in all_dense:
        W, b = layer.get_weights()
        in_sz, out_sz = W.shape
        name = layer.name.lower()

        f.write(f"/* {layer.name} [{in_sz} -> {out_sz}] */\n")
        f.write(f"static const float {name}_W[{out_sz}][{in_sz}] = {{\n")
        for o in range(out_sz):
            row = ", ".join([f"{W[i, o]:.8f}f" for i in range(in_sz)])
            comma = "," if o < out_sz - 1 else ""
            f.write(f"    {{ {row} }}{comma}\n")
        f.write("};\n\n")

        f.write(f"static const float {name}_b[{out_sz}] = {{\n    ")
        f.write(", ".join([f"{v:.8f}f" for v in b]))
        f.write("\n};\n\n")

    f.write("static inline void nn_inference(\n")
    f.write("    const float input[NN_INPUT_SIZE],\n")
    f.write("    float *d1_out,\n")
    f.write("    float *d2_out)\n")
    f.write("{\n")
    f.write("    float buf_a[NN_MAX_HIDDEN];\n")
    f.write("    float buf_b[NN_MAX_HIDDEN];\n")
    f.write("    uint32_t i, j;\n\n")

    f.write("    for (i = 0U; i < NN_INPUT_SIZE; i++) {\n")
    f.write("        buf_a[i] = (input[i] - NN_MEAN[i]) / NN_STD[i];\n")
    f.write("    }\n\n")

    for layer in shared_dense:
        W, b = layer.get_weights()
        in_sz, out_sz = W.shape
        name = layer.name.lower()

        f.write(f"    /* {layer.name} */\n")
        f.write(f"    for (i = 0U; i < {out_sz}U; i++) {{\n")
        f.write(f"        buf_b[i] = {name}_b[i];\n")
        f.write(f"        for (j = 0U; j < {in_sz}U; j++) {{\n")
        f.write(f"            buf_b[i] += {name}_W[i][j] * buf_a[j];\n")
        f.write("        }\n")
        f.write("        if (buf_b[i] < 0.0f) buf_b[i] = 0.0f;\n")
        f.write("    }\n")
        f.write(f"    for (i = 0U; i < {out_sz}U; i++) buf_a[i] = buf_b[i];\n\n")

    out_vars = ["d1_out", "d2_out"]
    mins = ["NN_D1_FEAT_MIN", "NN_D2_FEAT_MIN"]
    maxs = ["NN_D1_FEAT_MAX", "NN_D2_FEAT_MAX"]

    for layer, out_var, mn, mx in zip(output_dense, out_vars, mins, maxs):
        W, b = layer.get_weights()
        in_sz, out_sz = W.shape
        name = layer.name.lower()

        f.write(f"    /* {layer.name} */\n")
        f.write("    {\n")
        f.write(f"        float raw = {name}_b[0];\n")
        f.write(f"        for (j = 0U; j < {in_sz}U; j++) raw += {name}_W[0][j] * buf_a[j];\n")
        f.write("        raw = 1.0f / (1.0f + expf(-raw));\n")
        f.write(f"        raw = raw * ({mx} - {mn}) + {mn};\n")
        f.write("        if (raw < D_MIN) raw = D_MIN;\n")
        f.write("        if (raw > D_MAX) raw = D_MAX;\n")
        f.write(f"        *{out_var} = raw;\n")
        f.write("    }\n\n")

    f.write("}\n\n")
    f.write("#endif\n")

print(f"Header file generated: {OUTPUT_FILE}")

# =============================================================================
# 5. Summary
# =============================================================================
total_params = sum(np.prod(w.shape) for l in all_dense for w in l.get_weights())
print("=" * 65)
print("Header Summary:")
print(f"Dense layers: {len(all_dense)}")
print(f"Total parameters: {total_params:,}")
print(f"Flash usage: {total_params * 4 / 1024:.1f} KB")
print(f"RAM usage: {max_hidden * 2 * 4} bytes")
print("=" * 65)
