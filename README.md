<p align="center">
  <img src="https://raw.githubusercontent.com/SRRasaei/cascaded-boost-ml/main/.github/banner_minimal_tech.svg" alt="RezaSeek Minimal Tech Banner"/>
</p>
<svg width="1200" height="260" viewBox="0 0 1200 260" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#0f0f0f"/>
      <stop offset="100%" stop-color="#1a1a1a"/>
    </linearGradient>
    <linearGradient id="accent" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#00e0ff"/>
      <stop offset="100%" stop-color="#00ffa6"/>
    </linearGradient>
  </defs>

  <rect width="1200" height="260" fill="url(#grad)" rx="14"/>

  <text x="50%" y="45%" fill="white" font-size="48" font-family="Segoe UI, Roboto, sans-serif" text-anchor="middle" letter-spacing="2">
    RezaSeek Cascaded Boost ML
  </text>

  <text x="50%" y="68%" fill="url(#accent)" font-size="22" font-family="Segoe UI, Roboto, sans-serif" text-anchor="middle" letter-spacing="3">
    Minimal Neural Controller for Dual Boost Converter
  </text>

  <line x1="300" y1="190" x2="900" y2="190" stroke="url(#accent)" stroke-width="1.2" opacity="0.4"/>
</svg>



# Cascaded Boost ML  
Machine‑Learning Controller for a Dual Cascaded Boost Converter (STM32G431RB)

This repository contains the complete pipeline for generating datasets, training neural models, and exporting MCU‑ready C headers for a dual cascaded boost converter.  
The system is designed for real‑time control on **STM32G431RB** using a lightweight neural network optimized for embedded deployment.

---

## 🚀 Features

- Physics‑based dataset generator (steady‑state + transient)
- Iterative boost‑stage solver with energy‑consistent modeling
- Noise‑augmented measurements simulating ADC + sensor behavior
- Neural controller optimized for STM32 (no BN, no Dropout)
- Automatic export of C header (`cascaded_model_weights.h`)
- TFLite model export for validation
- Reproducible seeds for dataset and transient generation

---

## 📂 Project Structure

cascaded-boost-ml/
├─ src/
│  ├─ cascaded_dataset_generator.py
│  ├─ train_cascaded_model.py
│  └─ extract_cascaded_weights.py
│
├─ data/
│  └─ cascaded_boost_dataset_final.csv
│
├─ models/
│  ├─ cascaded_boost_model.h5
│  ├─ cascaded_boost_model.tflite
│  └─ cascaded_model_weights.h
│
├─ docs/
│  └─ model_summary.txt
│
├─ requirements.txt
├─ LICENSE
└─ README.md


---

## 📦 Installation

### 1. Create a virtual environment
```bash
python -m venv .venv

2. Activate it
Windows PowerShell
.venv\Scripts\Activate.ps1

Linux / macOS
source .venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

🧪 Dataset Generation
Run the dataset generator:
python src/cascaded_dataset_generator.py

Output:
data/cascaded_boost_dataset_final.csv

The generator includes:

Extended duty‑cycle ranges

Energy‑consistent boost‑stage solver

ADC‑like noise injection

Steady‑state + transient samples

Automatic filtering of invalid samples

🧠 Model Training
Train the neural controller:
python src/train_cascaded_model.py

Outputs:
models/cascaded_boost_model.h5
models/cascaded_boost_model.tflite
scaler_params_cascaded.npz

Model architecture:
Input(7)
 → Dense(64, ReLU)
 → Dense(32, ReLU)
 → Dense(16, ReLU)
 → [Output_D1(sigmoid), Output_D2(sigmoid)]

Optimized for:

Flash ≈ 12.3 KB

RAM ≈ 512 bytes

Inference time ≈ 36 µs @ 170 MHz (FPU enabled)

🛠 Exporting C Header for STM32
Generate the MCU‑ready header:
python src/extract_cascaded_weights.py

Output:
models/cascaded_model_weights.h

This header includes:

All Dense layer weights and biases

Input normalization constants

Output scaling to duty‑cycle range [0.05, 0.92]

Fully deterministic inference function nn_inference()

🔧 STM32 Integration
1. Copy the file:
models/cascaded_model_weights.h

into:
Core/Inc/

2. Use the controller in your firmware:
#include "cascaded_model_weights.h"

float inputs[7] = {
    Vin, Vint, Iint, Vout, Iout, Vref, Vout - Vref
};

float d1, d2;
nn_inference(inputs, &d1, &d2);

3. Apply d1 and d2 to your PWM timers.

📊 Documentation
Model summary is available in:
docs/model_summary.txt

Includes:

Layer shapes

Parameter counts

Flash/RAM usage

Inference timing

Scaling details

📄 License
This project is released under the MIT License.
See the LICENSE file for details.

⭐ Acknowledgements
This project is part of the RezaSeek embedded ML research initiative, combining power electronics, real‑time control, and neural inference on microcontrollers.
📬 Contact
For questions or collaboration:

Seyed Reza Rasaie  
GitHub: https://github.com/SRRasaei


---

If you want, I can also generate:

✅ `model_summary.txt`  
✅ `requirements.txt`  
✅ `LICENSE` (MIT)  
✅ Folder structure commands  
✅ Release notes for GitHub  
✅ A banner or badges for the README  

Just tell me what you want next.




