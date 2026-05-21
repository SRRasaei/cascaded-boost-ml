"""
===============================================================================
Cascaded Boost Converter Dataset Generator — Version 1.0.0
Author: Seyed Reza Rasaei 

Description:
    Physics-based dataset generator for a dual-stage cascaded boost converter.

    Features:
      - Iterative, energy-consistent boost stage solver
      - MOSFET conduction and switching losses
      - Diode and inductor losses
      - ADC-like measurement noise on all variables
      - Extended duty-cycle range [0.05, 0.88] (aligned with firmware limits)
      - Separate seeds for steady-state and transient samples
      - Filtering of unreachable Vref values
      - Basic efficiency statistics in the final report

Output:
    cascaded_boost_dataset_final.csv
===============================================================================
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


# =============================================================================
# Cascaded Boost Converter Simulator
# =============================================================================
class CascadedBoostSimulator:
    """
    Physics-based simulator for a dual-stage cascaded boost converter.
    Includes MOSFET losses, diode losses, inductor losses, and switching losses.
    """

    def __init__(self):
        # Stage 1 parameters
        self.L1 = 100e-6
        self.C1 = 220e-6
        self.Rds1 = 0.05
        self.Vf1 = 0.7
        self.RL1 = 0.1

        # Stage 2 parameters
        self.L2 = 100e-6
        self.C2 = 470e-6
        self.Rds2 = 0.05
        self.Vf2 = 0.7
        self.RL2 = 0.1

        # Switching frequency (aligned with hardware)
        # PSC=169, ARR=333, CLK=170MHz → f_sw ≈ 30 kHz
        self.f_sw = 30_000

        # MOSFET switching transition time (t_rise + t_fall) / 2
        self.t_sw = 20e-9  # 20 ns

    # -------------------------------------------------------------------------
    def boost_stage(
        self,
        Vin: float,
        D: float,
        R_load: float,
        Rds: float,
        Vf: float,
        RL: float,
    ) -> float:
        """
        Compute the output voltage of a single boost stage using an iterative,
        energy-consistent method.

        Compared to naive ideal formulas, this approach:
          - Uses Iin derived from Iout and duty (Iin = Iout / (1 - D))
          - Computes input power and subtracts detailed losses
          - Iterates until Vout converges

        Args:
            Vin (float): Input voltage (V)
            D (float): Duty cycle [0.05, 0.92]
            R_load (float): Load resistance (Ohm)
            Rds (float): MOSFET on-resistance (Ohm)
            Vf (float): Diode forward voltage (V)
            RL (float): Inductor series resistance (Ohm)

        Returns:
            float: Realistic output voltage after convergence (V)
        """
        D = np.clip(D, 0.05, 0.92)
        Vin = max(Vin, 0.1)

        # Ideal upper bound
        Vout_ideal = Vin / (1.0 - D)

        # Initial guess: assume ~90% efficiency
        Vout_est = Vout_ideal * 0.90

        # Iterative solution (typically converges in 4–5 iterations)
        for _ in range(8):
            Iout_est = Vout_est / R_load

            # Input current from energy conservation: Iin = Iout / (1 - D)
            Iin_avg = Iout_est / (1.0 - D)

            # MOSFET conduction loss: I^2 * Rds * D
            P_mosfet_cond = (Iin_avg ** 2) * Rds * D

            # MOSFET switching loss: 0.5 * Vin * Iin * t_sw * f_sw
            P_mosfet_sw = 0.5 * Vin * Iin_avg * self.t_sw * self.f_sw

            # Diode loss: Vf * Iout
            P_diode = Vf * Iout_est

            # Inductor copper loss: I^2 * RL
            P_inductor = (Iin_avg ** 2) * RL

            # Input and output power
            P_in = Vin * Iin_avg
            P_loss = P_mosfet_cond + P_mosfet_sw + P_diode + P_inductor
            P_out = max(P_in - P_loss, 0.0)

            # Efficiency
            efficiency = np.clip(
                P_out / P_in if P_in > 1e-9 else 0.85,
                0.65,
                0.98,
            )

            Vout_new = Vout_ideal * efficiency

            # Convergence check
            if abs(Vout_new - Vout_est) < 1e-3:
                break

            Vout_est = Vout_new

        return Vout_est

    # -------------------------------------------------------------------------
    def vout_max(self, Vin: float, D_max: float = 0.88) -> float:
        """
        Conservative estimate of maximum achievable output voltage for
        two cascaded stages. Used to filter unreachable Vref values.

        Args:
            Vin (float): Input voltage (V)
            D_max (float): Maximum duty cycle used for estimation

        Returns:
            float: Maximum achievable output voltage (V)
        """
        V1 = Vin / (1.0 - D_max) * 0.88
        V2 = V1 / (1.0 - D_max) * 0.88
        return V2

    # -------------------------------------------------------------------------
    def simulate_steady_state(
        self,
        Vin: float,
        D1: float,
        D2: float,
        R_load: float,
    ):
        """
        Steady-state simulation of the cascaded boost converter.

        The second stage is seen by the first stage as an equivalent load:
            R_load_equiv = R_load / (1 - D2)^2

        Returns:
            tuple: (Vint, Vout, Iint, Iout) with ADC-like noise added.
        """
        Vin = np.clip(Vin, 5.0, 32.0)
        D1 = np.clip(D1, 0.05, 0.92)
        D2 = np.clip(D2, 0.05, 0.92)

        # Equivalent load seen by stage 1
        denom = (1.0 - D2) ** 2
        if denom > 0.01:
            R_load_equiv = R_load / denom
        else:
            R_load_equiv = R_load * 100.0

        # Stage 1
        Vint = self.boost_stage(
            Vin,
            D1,
            max(R_load_equiv, 1.0),
            self.Rds1,
            self.Vf1,
            self.RL1,
        )

        # Stage 2
        Vout = self.boost_stage(
            Vint,
            D2,
            R_load,
            self.Rds2,
            self.Vf2,
            self.RL2,
        )

        # DC currents
        Iout = Vout / R_load if R_load > 0 else 0.0
        Iint = Iout / (1.0 - D2 + 1e-6)

        # ADC-like measurement noise
        # Voltage: ~0.4% relative error + 5 mV offset
        # Current: ~0.8% relative error + 1 mA offset
        Vint += np.random.normal(0.0, 0.004 * abs(Vint) + 0.005)
        Vout += np.random.normal(0.0, 0.004 * abs(Vout) + 0.005)
        Iint += np.random.normal(0.0, 0.008 * abs(Iint) + 0.001)
        Iout += np.random.normal(0.0, 0.008 * abs(Iout) + 0.001)

        return Vint, Vout, Iint, Iout

    # -------------------------------------------------------------------------
    def simulate_transient(
        self,
        Vin: float,
        D1: float,
        D2: float,
        R_load: float,
        Vout_prev: float,
        alpha: float = 0.15,
    ):
        """
        Approximate transient simulation using a first-order response model.

        Args:
            Vin (float): Input voltage (V)
            D1 (float): Duty cycle of stage 1
            D2 (float): Duty cycle of stage 2
            R_load (float): Load resistance (Ohm)
            Vout_prev (float): Previous output voltage (V)
            alpha (float): Convergence factor (0 = slow, 1 = instant)

        Returns:
            tuple: (Vint_t, Vout_t, Iint_t, Iout_t) with noise added.
        """
        Vint_ss, Vout_ss, Iint_ss, _ = self.simulate_steady_state(
            Vin,
            D1,
            D2,
            R_load,
        )

        # First-order response for Vout
        Vout_t = Vout_prev + alpha * (Vout_ss - Vout_prev)
        Iout_t = Vout_t / R_load if R_load > 0 else 0.0

        # Vint assumed close to steady-state (faster dynamics)
        Vint_t = Vint_ss

        # Intermediate current proportional to transient output current
        Iint_t = Iout_t / (1.0 - D2 + 1e-6)

        # Noise on all four variables
        Vint_t += np.random.normal(0.0, 0.004 * abs(Vint_t) + 0.005)
        Vout_t += np.random.normal(0.0, 0.004 * abs(Vout_t) + 0.005)
        Iint_t += np.random.normal(0.0, 0.008 * abs(Iint_t) + 0.001)
        Iout_t += np.random.normal(0.0, 0.008 * abs(Iout_t) + 0.001)

        return Vint_t, Vout_t, Iint_t, Iout_t


# =============================================================================
# Dataset Generation
# =============================================================================

sim = CascadedBoostSimulator()

# Extended parameter ranges
Vin_range = np.arange(10, 31, 1)  # 10 V to 30 V
D1_range = np.arange(0.05, 0.89, 0.04)
D2_range = np.arange(0.05, 0.89, 0.04)
R_load_range = [20, 30, 50, 100, 200]
Vref_range = [20, 36, 48, 72, 96, 120, 150, 180]

print("=" * 65)
print("Cascaded Boost Dataset Generation — Version 1.0.0")
print(f"Vin range   : {Vin_range[0]} .. {Vin_range[-1]} V  ({len(Vin_range)} values)")
print(f"D1 range    : {D1_range[0]:.2f} .. {D1_range[-1]:.2f}  ({len(D1_range)} values)")
print(f"D2 range    : {D2_range[0]:.2f} .. {D2_range[-1]:.2f}  ({len(D2_range)} values)")
print(f"R_load list : {R_load_range}")
print(f"Vref list   : {Vref_range}")
print(f"f_sw        : {sim.f_sw:,} Hz")
print("=" * 65)

dataset = []
skipped_ss = 0

# -----------------------------------------------------------------------------
# Part 1: Steady-State Samples
# -----------------------------------------------------------------------------
print("\n[1/2] Generating steady-state samples...")

np.random.seed(42)

for Vin in tqdm(Vin_range, desc="Vin"):
    vout_max_possible = sim.vout_max(Vin)

    for R_load in R_load_range:
        for Vref in Vref_range:

            # Filter unreachable Vref values
            if Vref > vout_max_possible * 0.95:
                skipped_ss += 1
                continue

            for D1 in D1_range:
                for D2 in D2_range:
                    Vint, Vout, Iint, Iout = sim.simulate_steady_state(
                        Vin,
                        D1,
                        D2,
                        R_load,
                    )

                    error_Vout = Vout - Vref

                    dataset.append(
                        [
                            float(Vin),
                            Vint,
                            Iint,
                            Vout,
                            Iout,
                            float(Vref),
                            error_Vout,
                            D1,
                            D2,
                        ]
                    )

n_steady = len(dataset)
print(f"Steady-state samples      : {n_steady:,}")
print(f"Filtered (high Vref)      : {skipped_ss:,}")

# -----------------------------------------------------------------------------
# Part 2: Transient Samples
# -----------------------------------------------------------------------------
print("\n[2/2] Generating transient samples...")

np.random.seed(123)

n_transient_target = 20_000
transient_added = 0
skipped_trans = 0

for _ in tqdm(range(n_transient_target), desc="Transient"):
    Vin = np.random.uniform(10.0, 30.0)
    D1 = np.random.uniform(0.05, 0.88)
    D2 = np.random.uniform(0.05, 0.88)
    R_load = float(np.random.choice(R_load_range))
    Vref = float(np.random.choice(Vref_range))

    # Filter unreachable Vref values
    if Vref > sim.vout_max(Vin) * 0.95:
        skipped_trans += 1
        continue

    # Previous output voltage with random deviation from steady-state
    _, Vout_ss, _, _ = sim.simulate_steady_state(Vin, D1, D2, R_load)
    delta = np.random.uniform(-0.40, 0.40) * Vout_ss
    Vout_prev = max(Vout_ss + delta, 0.5)

    alpha = np.random.uniform(0.05, 0.50)

    Vint, Vout, Iint, Iout = sim.simulate_transient(
        Vin,
        D1,
        D2,
        R_load,
        Vout_prev,
        alpha,
    )

    error_Vout = Vout - Vref

    dataset.append(
        [
            Vin,
            Vint,
            Iint,
            Vout,
            Iout,
            Vref,
            error_Vout,
            D1,
            D2,
        ]
    )
    transient_added += 1

print(f"Transient samples added    : {transient_added:,}")
print(f"Filtered (high Vref)      : {skipped_trans:,}")

# -----------------------------------------------------------------------------
# Convert to DataFrame, Shuffle, Filter, Save
# -----------------------------------------------------------------------------
columns = [
    "Vin",
    "Vint",
    "Iint",
    "Vout",
    "Iout",
    "Vref",
    "error_Vout",
    "D1",
    "D2",
]
df = pd.DataFrame(dataset, columns=columns)

# Shuffle with fixed seed
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Remove rows with unrealistic values
before = len(df)
df = df[
    (df["Vout"] > 0.5)
    & (df["Vint"] > 0.5)
    & (df["Vout"] < 300)
    & (df["Vint"] < 300)
    & (df["D1"] >= 0.04)
    & (df["D2"] >= 0.04)
].reset_index(drop=True)
removed = before - len(df)

if removed > 0:
    print(f"Removed invalid rows       : {removed}")

output_file = "cascaded_boost_dataset_final.csv"
df.to_csv(output_file, index=False)

# -----------------------------------------------------------------------------
# Final Statistics
# -----------------------------------------------------------------------------
print("\n" + "=" * 65)
print("Final Dataset Statistics — Version 1.0.0")
print("=" * 65)
print(f"{'Total samples':<22}: {len(df):,}")
print(f"{'  → steady-state':<22}: {n_steady:,}")
print(f"{'  → transient':<22}: {transient_added:,}")
print(f"{'Vin':<22}: [{df['Vin'].min():.1f}, {df['Vin'].max():.1f}] V")
print(f"{'Vint':<22}: [{df['Vint'].min():.1f}, {df['Vint'].max():.1f}] V")
print(f"{'Vout':<22}: [{df['Vout'].min():.1f}, {df['Vout'].max():.1f}] V")
print(f"{'Iout':<22}: [{df['Iout'].min():.3f}, {df['Iout'].max():.3f}] A")
print(
    f"{'error_Vout':<22}: "
    f"[{df['error_Vout'].min():.1f}, {df['error_Vout'].max():.1f}] V"
)
print(f"{'D1':<22}: [{df['D1'].min():.3f}, {df['D1'].max():.3f}]")
print(f"{'D2':<22}: [{df['D2'].min():.3f}, {df['D2'].max():.3f}]")

# Approximate efficiency statistics
df["P_out"] = df["Vout"] * df["Iout"]
df["P_in"] = df["Vin"] * (
    df["Iout"] / (1 - df["D2"].clip(0.05, 0.92))
)
df["eta"] = (df["P_out"] / df["P_in"].clip(lower=0.1)).clip(0, 1)
print(
    f"{'Efficiency (approx)':<22}: "
    f"[{df['eta'].quantile(0.05):.3f}, "
    f"{df['eta'].quantile(0.95):.3f}]  (5th–95th percentile)"
)

print("-" * 65)
print(f"Dataset file '{output_file}' saved successfully.")
print("Next step: run train_closed_loop_model.py (or the appropriate training script).")
