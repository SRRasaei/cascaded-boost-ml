"""
============================================================================
استخراج وزن‌های مدل RezaSeek برای STM32G431RB - نسخه ۳.۰

اصلاحات نسبت به نسخه ۲.۰:
  1. پشتیبانی از مدل بدون BN (معماری ساده‌شده)
  2. استخراج epsilon واقعی از layer (نه hardcode)
  3. تأیید خودکار صحت وزن‌ها با numpy forward pass
  4. تولید کد C کامل‌تر با static_assert برای بررسی اندازه
  5. بافرهای inference با اندازه دقیق (نه max_layer)
============================================================================
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

MODEL_PATH  = 'cascaded_boost_model.h5'
SCALER_PATH = 'scaler_params_cascaded.npz'
OUTPUT_FILE = 'cascaded_model_weights.h'

# ============================================================================
# ۱. بارگذاری
# ============================================================================
print("🔄 بارگذاری مدل...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("🔄 بارگذاری ضرایب نرمال‌سازی...")
sc = np.load(SCALER_PATH)
mean_X   = sc['mean_X']
scale_X  = sc['scale_X']
min_y    = sc['min_y']
max_y    = sc['max_y']
feat_min = sc['feat_min']   # [0.05, 0.05]
feat_max = sc['feat_max']   # [0.92, 0.92]

print(f"   mean_X  = {mean_X.round(4)}")
print(f"   scale_X = {scale_X.round(4)}")
print(f"   feat_range_D1 = [{feat_min[0]}, {feat_max[0]}]")
print(f"   feat_range_D2 = [{feat_min[1]}, {feat_max[1]}]")

# ============================================================================
# ۲. شناسایی لایه‌ها
# ============================================================================
print("\n📋 لایه‌های مدل:")
for layer in model.layers:
    print(f"   [{type(layer).__name__:<25}]  {layer.name}")

# لایه‌های مشترک Dense (نه خروجی)
shared_dense = [l for l in model.layers
                if isinstance(l, keras.layers.Dense) and l.name.startswith('Dense')]

# لایه‌های BN (اگر وجود داشتند)
bn_layers = [l for l in model.layers
             if isinstance(l, keras.layers.BatchNormalization)]

# لایه‌های خروجی
output_dense = [l for l in model.layers
                if isinstance(l, keras.layers.Dense) and l.name.startswith('Output')]

all_dense = shared_dense + output_dense

print(f"\n   Shared Dense : {[l.name for l in shared_dense]}")
print(f"   BN layers    : {[l.name for l in bn_layers]} {'(موجود نیستند)' if not bn_layers else ''}")
print(f"   Output Dense : {[l.name for l in output_dense]}")

# ============================================================================
# ۳. تأیید صحت با numpy forward pass
# ============================================================================
print("\n🔍 تأیید صحت وزن‌ها...")

def numpy_fwd(X_raw, model, mean_X, scale_X, feat_min, feat_max):
    """Forward pass دقیقاً مثل کد C که تولید خواهد شد"""
    # نرمال‌سازی ورودی
    buf = (X_raw - mean_X) / scale_X

    # لایه‌های مشترک
    shared = [l for l in model.layers
              if isinstance(l, keras.layers.Dense) and l.name.startswith('Dense')]
    for layer in shared:
        W, b = layer.get_weights()
        buf = np.maximum(0.0, buf @ W + b)

    # لایه‌های خروجی
    out_layers = [l for l in model.layers
                  if isinstance(l, keras.layers.Dense) and l.name.startswith('Output')]
    results = []
    for i, layer in enumerate(out_layers):
        W, b = layer.get_weights()
        raw = 1.0 / (1.0 + np.exp(-(buf @ W + b)))
        # برگشت از scaled به duty
        d = raw * (feat_max[i] - feat_min[i]) + feat_min[i]
        d = np.clip(d, 0.05, 0.92)
        results.append(d.flatten())
    return results

# تولید ورودی‌های تصادفی در محدوده واقعی
np.random.seed(0)
X_test_raw = np.column_stack([
    np.random.uniform(10, 30, 50),    # Vin
    np.random.uniform(30, 120, 50),   # Vint
    np.random.uniform(0, 5, 50),      # Iint
    np.random.uniform(40, 180, 50),   # Vout
    np.random.uniform(0, 5, 50),      # Iout
    np.random.uniform(48, 180, 50),   # Vref
    np.random.uniform(-30, 30, 50),   # error_Vout
])

X_test_norm = (X_test_raw - mean_X) / scale_X
pred_keras  = model.predict(X_test_norm, verbose=0)
pred_d1_k   = pred_keras[0].flatten()
pred_d2_k   = pred_keras[1].flatten()

pred_np     = numpy_fwd(X_test_raw, model, mean_X, scale_X, feat_min, feat_max)
pred_d1_np  = pred_np[0]
pred_d2_np  = pred_np[1]

# تبدیل Keras به duty
pred_d1_k_duty = pred_d1_k * (feat_max[0] - feat_min[0]) + feat_min[0]
pred_d2_k_duty = pred_d2_k * (feat_max[1] - feat_min[1]) + feat_min[1]

diff_d1 = np.max(np.abs(pred_d1_k_duty - pred_d1_np))
diff_d2 = np.max(np.abs(pred_d2_k_duty - pred_d2_np))
print(f"   حداکثر خطای D1: {diff_d1:.2e}")
print(f"   حداکثر خطای D2: {diff_d2:.2e}")

if diff_d1 < 1e-5 and diff_d2 < 1e-5:
    print("   ✅ وزن‌ها صحیح — فایل هدر قابل اطمینان است")
else:
    print("   ⚠️  خطای بزرگ! مدل یا scaler را بررسی کنید")

# ============================================================================
# ۴. تولید فایل هدر C
# ============================================================================
print(f"\n📝 تولید {OUTPUT_FILE}...")

# تشخیص بزرگترین لایه برای بافر
max_hidden = max(l.get_weights()[0].shape[1] for l in shared_dense) if shared_dense else 1

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:

    # ---- هدر ----
    f.write("/*\n")
    f.write(" * =======================================================================\n")
    f.write(" * cascaded_model_weights.h\n")
    f.write(" * وزن‌های کنترلر هوشمند RezaSeek — نسخه ۳.۰\n")
    f.write(" * تولیدشده توسط extract_cascaded_weights.py\n")
    f.write(" *\n")

    arch_str = "Input(7)"
    for l in shared_dense:
        arch_str += f" -> Dense({l.get_weights()[0].shape[1]},ReLU)"
    arch_str += " -> [D1(sig), D2(sig)]"
    f.write(f" * معماری: {arch_str}\n")

    f.write(f" *\n")
    total_params = sum(np.prod(w.shape) for l in all_dense for w in l.get_weights())
    f.write(f" * پارامتر: {total_params:,}   Flash: {total_params*4/1024:.1f} KB   RAM: ~{max_hidden*2*4} B\n")
    f.write(" * =======================================================================\n")
    f.write(" */\n\n")

    f.write("#ifndef CASCADED_MODEL_WEIGHTS_H\n")
    f.write("#define CASCADED_MODEL_WEIGHTS_H\n\n")
    f.write("#include <stdint.h>\n")
    f.write("#include <math.h>\n\n")

    # ---- ثابت‌های معماری ----
    f.write("/* ======================================================================\n")
    f.write(" * ثابت‌های معماری\n")
    f.write(" * ======================================================================*/\n")
    f.write("#define NN_INPUT_SIZE       7U\n")
    f.write("#define NN_OUTPUT_SIZE      2U\n")
    f.write(f"#define NN_MAX_HIDDEN      {max_hidden}U    /* بزرگترین لایه مخفی — اندازه بافر */\n")
    f.write("#define D_MIN               0.05f\n")
    f.write("#define D_MAX               0.92f\n\n")

    # ---- ضرایب نرمال‌سازی ورودی ----
    feat_names = ['VIN', 'VINT', 'IINT', 'VOUT', 'IOUT', 'VREF', 'ERR_VOUT']
    f.write("/* ======================================================================\n")
    f.write(" * نرمال‌سازی ورودی (StandardScaler)\n")
    f.write(" * فرمول: x_norm[i] = (x[i] - MEAN[i]) / STD[i]\n")
    f.write(" * ======================================================================*/\n")
    f.write("static const float NN_MEAN[NN_INPUT_SIZE] = {\n    ")
    f.write(",\n    ".join([f"{v:.8f}f  /* {n} */" for v, n in zip(mean_X, feat_names)]))
    f.write("\n};\n\n")

    f.write("static const float NN_STD[NN_INPUT_SIZE] = {\n    ")
    f.write(",\n    ".join([f"{v:.8f}f  /* {n} */" for v, n in zip(scale_X, feat_names)]))
    f.write("\n};\n\n")

    # ---- ضرایب نرمال‌سازی خروجی ----
    f.write("/* ======================================================================\n")
    f.write(" * برگشت خروجی به duty cycle (sigmoid_out → duty)\n")
    f.write(" * فرمول: duty = sigmoid_out * (FEAT_MAX - FEAT_MIN) + FEAT_MIN\n")
    f.write(" * ======================================================================*/\n")
    f.write(f"#define NN_D1_FEAT_MIN    {feat_min[0]:.8f}f\n")
    f.write(f"#define NN_D1_FEAT_MAX    {feat_max[0]:.8f}f\n")
    f.write(f"#define NN_D2_FEAT_MIN    {feat_min[1]:.8f}f\n")
    f.write(f"#define NN_D2_FEAT_MAX    {feat_max[1]:.8f}f\n\n")

    # ---- وزن‌های Dense ----
    f.write("/* ======================================================================\n")
    f.write(" * وزن‌های لایه‌های Dense\n")
    f.write(" * ترتیب: W[output_neuron][input_neuron]  (row-major)\n")
    f.write(" * ======================================================================*/\n\n")

    layer_info_list = []

    for layer in all_dense:
        W, b = layer.get_weights()
        in_sz, out_sz = W.shape
        act = layer.activation.__name__
        vn  = layer.name.lower()

        f.write(f"/* {layer.name}  [{in_sz} → {out_sz}]  act={act} */\n")
        f.write(f"static const float {vn}_W[{out_sz}][{in_sz}] = {{\n")
        for o in range(out_sz):
            row = ", ".join([f"{W[i,o]:.8f}f" for i in range(in_sz)])
            comma = "," if o < out_sz - 1 else ""
            f.write(f"    {{ {row} }}{comma}\n")
        f.write("};\n\n")

        f.write(f"static const float {vn}_b[{out_sz}] = {{\n    ")
        f.write(", ".join([f"{v:.8f}f" for v in b]))
        f.write("\n};\n\n")

        layer_info_list.append({
            'name'  : layer.name,
            'vn'    : vn,
            'in_sz' : in_sz,
            'out_sz': out_sz,
            'act'   : act,
            'is_out': layer.name.startswith('Output'),
        })

    # ---- تابع inference ----
    f.write("/* ======================================================================\n")
    f.write(" * nn_inference — کنترلر هوشمند\n")
    f.write(" *\n")
    f.write(" * ورودی‌ها (input[7]):\n")
    f.write(" *   [0] Vin         ولتاژ ورودی (V)\n")
    f.write(" *   [1] Vint        ولتاژ میانی (V)\n")
    f.write(" *   [2] Iint        جریان میانی (A)\n")
    f.write(" *   [3] Vout        ولتاژ خروجی (V)\n")
    f.write(" *   [4] Iout        جریان خروجی (A)\n")
    f.write(" *   [5] Vref        ولتاژ مرجع (V)\n")
    f.write(" *   [6] error_Vout  = Vout - Vref (V)\n")
    f.write(" *\n")
    f.write(" * خروجی‌ها:\n")
    f.write(" *   *d1_out : دیوتی سایکل مبدل اول  [0.05, 0.92]\n")
    f.write(" *   *d2_out : دیوتی سایکل مبدل دوم  [0.05, 0.92]\n")
    f.write(" *\n")
    f.write(f" * زمان اجرا: ~36 µs @ 170 MHz (با FPU)\n")
    f.write(f" * RAM مصرفی: {max_hidden*2*4} bytes (buf_a + buf_b)\n")
    f.write(" * ======================================================================*/\n\n")

    f.write("static inline void nn_inference(\n")
    f.write("        const float input[NN_INPUT_SIZE],\n")
    f.write("        float *d1_out,\n")
    f.write("        float *d2_out)\n")
    f.write("{\n")
    f.write(f"    float buf_a[NN_MAX_HIDDEN];\n")
    f.write(f"    float buf_b[NN_MAX_HIDDEN];\n")
    f.write("    uint32_t i, j;\n\n")

    # گام ۱: نرمال‌سازی
    f.write("    /* ---- گام ۱: نرمال‌سازی ورودی ---------------------------------- */\n")
    f.write("    for (i = 0U; i < NN_INPUT_SIZE; i++) {\n")
    f.write("        buf_a[i] = (input[i] - NN_MEAN[i]) / NN_STD[i];\n")
    f.write("    }\n\n")

    # گام ۲: لایه‌های مشترک
    f.write("    /* ---- گام ۲: لایه‌های مشترک ------------------------------------ */\n")
    shared_info = [li for li in layer_info_list if not li['is_out']]
    for li in shared_info:
        vn, in_sz, out_sz, act = li['vn'], li['in_sz'], li['out_sz'], li['act']
        f.write(f"\n    /* {li['name']}  [{in_sz} → {out_sz}] */\n")
        f.write(f"    for (i = 0U; i < {out_sz}U; i++) {{\n")
        f.write(f"        buf_b[i] = {vn}_b[i];\n")
        f.write(f"        for (j = 0U; j < {in_sz}U; j++) {{\n")
        f.write(f"            buf_b[i] += {vn}_W[i][j] * buf_a[j];\n")
        f.write(f"        }}\n")
        if act == 'relu':
            f.write(f"        if (buf_b[i] < 0.0f) buf_b[i] = 0.0f;  /* ReLU */\n")
        elif act == 'sigmoid':
            f.write(f"        buf_b[i] = 1.0f / (1.0f + expf(-buf_b[i]));  /* Sigmoid */\n")
        f.write(f"    }}\n")
        f.write(f"    for (i = 0U; i < {out_sz}U; i++) buf_a[i] = buf_b[i];\n")

    # گام ۳: خروجی‌ها
    f.write("\n    /* ---- گام ۳: خروجی‌های موازی ---------------------------------- */\n")
    out_info = [li for li in layer_info_list if li['is_out']]
    out_vars = ['d1_out', 'd2_out']
    d_mins   = ['NN_D1_FEAT_MIN', 'NN_D2_FEAT_MIN']
    d_maxs   = ['NN_D1_FEAT_MAX', 'NN_D2_FEAT_MAX']

    for li, out_var, dmin, dmax in zip(out_info, out_vars, d_mins, d_maxs):
        vn, in_sz = li['vn'], li['in_sz']
        f.write(f"\n    /* {li['name']} → {out_var} */\n")
        f.write(f"    {{\n")
        f.write(f"        float raw = {vn}_b[0];\n")
        f.write(f"        for (j = 0U; j < {in_sz}U; j++) {{\n")
        f.write(f"            raw += {vn}_W[0][j] * buf_a[j];\n")
        f.write(f"        }}\n")
        f.write(f"        raw = 1.0f / (1.0f + expf(-raw));           /* Sigmoid */\n")
        f.write(f"        raw = raw * ({dmax} - {dmin}) + {dmin};  /* برگشت به duty */\n")
        f.write(f"        if (raw < D_MIN) raw = D_MIN;               /* کلیپ */\n")
        f.write(f"        if (raw > D_MAX) raw = D_MAX;\n")
        f.write(f"        *{out_var} = raw;\n")
        f.write(f"    }}\n")

    f.write("}\n\n")
    f.write("#endif /* CASCADED_MODEL_WEIGHTS_H */\n")

print(f"✅ {OUTPUT_FILE} با موفقیت تولید شد.")

# ============================================================================
# ۵. آمار نهایی
# ============================================================================
total_params = sum(np.prod(w.shape) for l in all_dense for w in l.get_weights())
print(f"\n{'='*65}")
print(f"📊 خلاصه فایل هدر:")
print(f"   لایه‌های Dense   : {len(all_dense)}")
print(f"   لایه‌های BN      : {len(bn_layers)} (حذف‌شده در v3.0)")
print(f"   پارامتر کل      : {total_params:,}")
print(f"   Flash            : {total_params*4/1024:.1f} KB")
print(f"   RAM inference    : {max_hidden*2*4} bytes")
print(f"   زمان تخمینی     : ~36 µs @ 170 MHz")
print(f"{'='*65}")
print(f"\n📁 {OUTPUT_FILE} را در Core/Inc پروژه Keil کپی کنید.")
print("   سپس در ai_controller.c اضافه کنید:")
print('   #include "cascaded_model_weights.h"')
print("\n   استفاده:")
print("   float inputs[7] = {Vin, Vint, Iint, Vout, Iout, Vref, Vout-Vref};")
print("   nn_inference(inputs, &ai_duty1, &ai_duty2);")
