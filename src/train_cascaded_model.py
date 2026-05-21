"""
===============================================================================
Cascaded Boost Converter Neural Network Training — Version 1.0.0
Author: Seyed Reza Rasaei

Description:
    Trains a neural controller for a dual-stage cascaded boost converter.
    This version is optimized for STM32G431RB deployment.

Features:
    - Input vector: [Vin, Vint, Iint, Vout, Iout, Vref, error_Vout]
    - Output: [D1, D2]
    - No BatchNormalization (avoids inference ambiguity in C)
    - No Dropout (replaced with L2 regularization)
    - Lightweight architecture: 7 → 64 → 32 → 16 → [D1, D2]
    - MinMaxScaler with physical duty range (0.05–0.92)
    - NumPy inference verification before exporting weights
===============================================================================
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)
tf.random.set_seed(42)

# =============================================================================
# 1. Load Dataset
# =============================================================================
print("=" * 65)
print("Loading dataset...")
df = pd.read_csv("cascaded_boost_dataset_final.csv")
print(f"Total samples: {len(df)}")

if "error_Vout" not in df.columns:
    raise ValueError("Column 'error_Vout' is missing. Run cascaded_dataset_generator.py first.")

# =============================================================================
# 2. Select Input and Output Columns
# =============================================================================
FEATURE_COLS = ["Vin", "Vint", "Iint", "Vout", "Iout", "Vref", "error_Vout"]
TARGET_COLS = ["D1", "D2"]

X = df[FEATURE_COLS].values
y = df[TARGET_COLS].values

print(f"Input shape: {X.shape}   Output shape: {y.shape}")
print(f"D1 range: [{y[:,0].min():.3f}, {y[:,0].max():.3f}]")
print(f"D2 range: [{y[:,1].min():.3f}, {y[:,1].max():.3f}]")

# =============================================================================
# 3. Normalization
# =============================================================================
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler(feature_range=(0.05, 0.92))
scaler_y.fit(y)
y_scaled = scaler_y.transform(y)

print(f"\nInput scaler mean: {scaler_X.mean_.round(3)}")
print("Output scaler range: (0.05, 0.92)")

X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.15, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15, random_state=42
)

print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

# =============================================================================
# 4. Model Architecture (Optimized for STM32)
# =============================================================================
print("\nBuilding model (no BN, no Dropout)...")

reg = keras.regularizers.l2(1e-4)

inputs = keras.Input(shape=(7,), name="Inputs")
x = keras.layers.Dense(
    64, activation="relu", kernel_initializer="he_normal", kernel_regularizer=reg, name="Dense1"
)(inputs)
x = keras.layers.Dense(
    32, activation="relu", kernel_initializer="he_normal", kernel_regularizer=reg, name="Dense2"
)(x)
x = keras.layers.Dense(
    16, activation="relu", kernel_initializer="he_normal", kernel_regularizer=reg, name="Dense3"
)(x)

output_D1 = keras.layers.Dense(1, activation="sigmoid", name="Output_D1")(x)
output_D2 = keras.layers.Dense(1, activation="sigmoid", name="Output_D2")(x)

model = keras.Model(inputs=inputs, outputs=[output_D1, output_D2])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss={"Output_D1": "mse", "Output_D2": "mse"},
    loss_weights={"Output_D1": 1.0, "Output_D2": 1.0},
    metrics={"Output_D1": ["mae"], "Output_D2": ["mae"]},
)

model.summary()

total_params = model.count_params()
print(f"\nParameters: {total_params:,}   Flash: {total_params*4/1024:.1f} KB   RAM: ~512 B")

# =============================================================================
# 5. Callbacks
# =============================================================================
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=30, restore_best_weights=True, verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=12, min_lr=1e-6, verbose=1
)

checkpoint = keras.callbacks.ModelCheckpoint(
    "cascaded_boost_best.keras", monitor="val_loss", save_best_only=True, verbose=0
)

# =============================================================================
# 6. Training
# =============================================================================
print("\nTraining...")
history = model.fit(
    X_train,
    {"Output_D1": y_train[:, 0], "Output_D2": y_train[:, 1]},
    validation_data=(X_val, {"Output_D1": y_val[:, 0], "Output_D2": y_val[:, 1]}),
    epochs=500,
    batch_size=256,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1,
)

lr_key = "learning_rate" if "learning_rate" in history.history else "lr"
print(f"\nTraining stopped at epoch {len(history.history['loss'])}")

# =============================================================================
# 7. Evaluation
# =============================================================================
print("\nEvaluating on test set...")
y_pred_s = model.predict(X_test, verbose=0)
y_pred = scaler_y.inverse_transform(
    np.column_stack([y_pred_s[0].flatten(), y_pred_s[1].flatten()])
)
y_true = scaler_y.inverse_transform(y_test)
y_pred = np.clip(y_pred, 0.05, 0.92)

r2_d1 = r2_score(y_true[:, 0], y_pred[:, 0])
r2_d2 = r2_score(y_true[:, 1], y_pred[:, 1])
mae_d1 = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
mae_d2 = mean_absolute_error(y_true[:, 1], y_pred[:, 1])

print(f"D1: R²={r2_d1:.5f}  MAE={mae_d1:.5f}")
print(f"D2: R²={r2_d2:.5f}  MAE={mae_d2:.5f}")

# =============================================================================
# 8. NumPy Inference Verification
# =============================================================================
print("\nVerifying NumPy inference...")

def numpy_inference(X_in, model):
    dense_shared = [
        l for l in model.layers
        if isinstance(l, keras.layers.Dense) and l.name.startswith("Dense")
    ]
    dense_out = [
        l for l in model.layers
        if isinstance(l, keras.layers.Dense) and l.name.startswith("Output")
    ]

    buf = X_in.copy()
    for layer in dense_shared:
        W, b = layer.get_weights()
        buf = np.maximum(0.0, buf @ W + b)

    outputs = []
    for layer in dense_out:
        W, b = layer.get_weights()
        out = 1.0 / (1.0 + np.exp(-(buf @ W + b)))
        outputs.append(out.flatten())

    return np.column_stack(outputs)

n_check = min(200, len(X_test))
idx_check = np.random.choice(len(X_test), n_check, replace=False)

pred_keras_raw = model.predict(X_test[idx_check], verbose=0)
pred_keras_np = np.column_stack([pred_keras_raw[0].flatten(), pred_keras_raw[1].flatten()])
pred_numpy = numpy_inference(X_test[idx_check], model)

max_diff = np.max(np.abs(pred_keras_np - pred_numpy))
print(f"Max difference (Keras vs NumPy): {max_diff:.2e}")

if max_diff < 1e-5:
    print("NumPy verification passed.")
else:
    print("WARNING: Large mismatch detected.")

# =============================================================================
# 9. Plots
# =============================================================================
fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 3, figure=fig)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(history.history["loss"], label="Train")
ax1.plot(history.history["val_loss"], label="Val")
ax1.set_title("Total Loss")
ax1.set_xlabel("Epoch")
ax1.set_yscale("log")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(history.history["Output_D1_loss"], "--", label="D1 Train")
ax2.plot(history.history["val_Output_D1_loss"], "--", label="D1 Val")
ax2.plot(history.history["Output_D2_loss"], label="D2 Train")
ax2.plot(history.history["val_Output_D2_loss"], label="D2 Val")
ax2.set_title("Per-Output Loss")
ax2.set_xlabel("Epoch")
ax2.set_yscale("log")
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(history.history[lr_key], color="orange")
ax3.set_title("Learning Rate")
ax3.set_xlabel("Epoch")
ax3.set_yscale("log")
ax3.grid(True, alpha=0.3)

lims = [0.05, 0.92]
sp = np.random.choice(len(y_true), min(3000, len(y_true)), replace=False)

ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(y_true[sp, 0], y_pred[sp, 0], alpha=0.3, s=5)
ax4.plot(lims, lims, "r--")
ax4.set_title(f"D1  R²={r2_d1:.4f}  MAE={mae_d1:.4f}")
ax4.grid(True, alpha=0.3)

ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(y_true[sp, 1], y_pred[sp, 1], alpha=0.3, s=5)
ax5.plot(lims, lims, "r--")
ax5.set_title(f"D2  R²={r2_d2:.4f}  MAE={mae_d2:.4f}")
ax5.grid(True, alpha=0.3)

ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(y_pred[:, 0] - y_true[:, 0], bins=60, alpha=0.6, label="D1")
ax6.hist(y_pred[:, 1] - y_true[:, 1], bins=60, alpha=0.6, label="D2")
ax6.axvline(0, color="red", linestyle="--")
ax6.set_title("Error Distribution")
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.suptitle("RezaSeek Neural Controller — Version 1.0.0", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("cascaded_training_results.png", dpi=150, bbox_inches="tight")
plt.show()

# =============================================================================
# 10. Save Model and Scalers
# =============================================================================
print("\nSaving model and scalers...")
model.save("cascaded_boost_model.keras")
model.save("cascaded_boost_model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
with open("cascaded_boost_model.tflite", "wb") as fout:
    fout.write(converter.convert())

np.savez(
    "scaler_params_cascaded.npz",
    mean_X=scaler_X.mean_,
    scale_X=scaler_X.scale_,
    min_y=scaler_y.data_min_,
    max_y=scaler_y.data_max_,
    feat_min=np.array([0.05, 0.05]),
    feat_max=np.array([0.92, 0.92]),
    n_features=np.array([7]),
    n_outputs=np.array([2]),
)

print("All files saved successfully. Next step: run extract_cascaded_weights.py")
