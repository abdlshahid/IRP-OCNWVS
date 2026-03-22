import numpy as np
import os

# ============================================================================
# 1️⃣ LOAD PROCESSED DATA
# ============================================================================

print("=" * 70)
print("STEP 1: Loading processed data")
print("=" * 70)

processed_file = os.path.join("data", "processed", "era5_wave_2019_training.npz")

if not os.path.exists(processed_file):
    raise FileNotFoundError(f"Processed file not found: {processed_file}")

data = np.load(processed_file)
X = data["X"]
y_true = data["y"]
freq_hz = data["freq_hz"]
direction_deg = data["direction_deg"]
valid_time = data["valid_time"]

print(f"✓ Loaded: {processed_file}")
print(f"  X shape: {X.shape} (time, 5, 30)")
print(f"  y_true shape: {y_true.shape} (time, 30, 24)")
print(f"  Frequencies: {len(freq_hz)}")
print(f"  Directions: {len(direction_deg)}")
print(f"  Time steps: {X.shape[0]}")

# ============================================================================
# 2️⃣ EXTRACT INPUTS
# ============================================================================

print("\n" + "=" * 70)
print("STEP 2: Extracting inputs S_f, a1, b1, a2, b2")
print("=" * 70)

S_f = X[:, 0, :]  # (time, 30)
a1 = X[:, 1, :]   # (time, 30)
b1 = X[:, 2, :]   # (time, 30)
a2 = X[:, 3, :]   # (time, 30)
b2 = X[:, 4, :]   # (time, 30)

print(f"S_f shape: {S_f.shape}")
print(f"a1 shape: {a1.shape}")
print(f"b1 shape: {b1.shape}")
print(f"a2 shape: {a2.shape}")
print(f"b2 shape: {b2.shape}")

print(f"\nS_f range: [{np.nanmin(S_f):.6e}, {np.nanmax(S_f):.6e}]")
print(f"a1 range: [{np.nanmin(a1):.4f}, {np.nanmax(a1):.4f}]")
print(f"b1 range: [{np.nanmin(b1):.4f}, {np.nanmax(b1):.4f}]")
print(f"a2 range: [{np.nanmin(a2):.4f}, {np.nanmax(a2):.4f}]")
print(f"b2 range: [{np.nanmin(b2):.4f}, {np.nanmax(b2):.4f}]")

# ============================================================================
# 3️⃣ RECONSTRUCT SECOND HARMONIC SPECTRUM
# ============================================================================

print("\n" + "=" * 70)
print("STEP 3: Reconstructing spectrum using first + second harmonics")
print("=" * 70)

# Convert directions to radians
direction_rad = np.deg2rad(direction_deg)

# Direction spacing (radians)
ddir = np.deg2rad(15.0)  # ERA5 uses 15° spacing

# Extended spectrum: S_extended(f, θ) = S_f(f) × D(f, θ)
# where D(f, θ) = 1/(2π) × (1 + 2·a1·cos(θ) + 2·b1·sin(θ) + 2·a2·cos(2θ) + 2·b2·sin(2θ))

n_time = X.shape[0]
n_freq = 30
n_dir = 24

S_extended = np.zeros((n_time, n_freq, n_dir))

for t in range(n_time):
    for f in range(n_freq):
        S_f_val = S_f[t, f]
        a1_val = a1[t, f]
        b1_val = b1[t, f]
        a2_val = a2[t, f]
        b2_val = b2[t, f]

        # Extended directional distribution with first + second harmonics
        # D = 1/(2π) × (1 + 2·a1·cos(θ) + 2·b1·sin(θ) + 2·a2·cos(2θ) + 2·b2·sin(2θ))
        D = (1.0 / (2 * np.pi)) * (
            1.0 
            + 2 * a1_val * np.cos(direction_rad) 
            + 2 * b1_val * np.sin(direction_rad)
            + 2 * a2_val * np.cos(2 * direction_rad)
            + 2 * b2_val * np.sin(2 * direction_rad)
        )

        # Ensure D is non-negative
        D = np.maximum(D, 0.0)

        # Renormalize D so that ∫ D dθ = 1
        # Compute integral: sum(D) * ddir
        D_integral = np.sum(D) * ddir
        if D_integral > 1e-10:
            D = D / D_integral

        # S_extended(f, θ) = S_f(f) × D(f, θ)
        S_extended[t, f, :] = S_f_val * D

print(f"✓ Reconstructed extended spectrum")
print(f"  S_extended shape: {S_extended.shape} (time, 30, 24)")
print(f"  S_extended range: [{np.nanmin(S_extended):.6e}, {np.nanmax(S_extended):.6e}]")

# ============================================================================
# 4️⃣ COMPUTE ERROR
# ============================================================================

print("\n" + "=" * 70)
print("STEP 4: Computing reconstruction error")
print("=" * 70)

# Remove NaN values for error computation
y_true_clean = np.nan_to_num(y_true, nan=0.0)
S_extended_clean = np.nan_to_num(S_extended, nan=0.0)

# Relative L2 error per time-freq pair
# error = ||s_extended - s_true|| / ||s_true||
epsilon = 1e-10  # avoid division by zero

norm_true = np.linalg.norm(y_true_clean.reshape(n_time, -1), axis=1)
norm_diff = np.linalg.norm((S_extended_clean - y_true_clean).reshape(n_time, -1), axis=1)

relative_l2_error = norm_diff / (norm_true + epsilon)

# MSE per time step
mse_per_time = np.mean((S_extended_clean - y_true_clean) ** 2, axis=(1, 2))

# Mean Absolute Error per time step
mae_per_time = np.mean(np.abs(S_extended_clean - y_true_clean), axis=(1, 2))

print(f"✓ Error metrics computed")
print(f"\n  Relative L2 Error (per time step):")
print(f"    Mean: {np.mean(relative_l2_error):.6f}")
print(f"    Std:  {np.std(relative_l2_error):.6f}")
print(f"    Min:  {np.min(relative_l2_error):.6f}")
print(f"    Max:  {np.max(relative_l2_error):.6f}")

print(f"\n  MSE (per time step):")
print(f"    Mean: {np.mean(mse_per_time):.6e}")
print(f"    Std:  {np.std(mse_per_time):.6e}")
print(f"    Min:  {np.min(mse_per_time):.6e}")
print(f"    Max:  {np.max(mse_per_time):.6e}")

print(f"\n  MAE (per time step):")
print(f"    Mean: {np.mean(mae_per_time):.6e}")
print(f"    Std:  {np.std(mae_per_time):.6e}")
print(f"    Min:  {np.min(mae_per_time):.6e}")
print(f"    Max:  {np.max(mae_per_time):.6e}")

# ============================================================================
# 5️⃣ PRINT RESULTS
# ============================================================================

print("\n" + "=" * 70)
print("SECOND HARMONIC RECONSTRUCTION RESULTS")
print("=" * 70)

print(f"\nOverall Statistics:")
print(f"  Total time steps: {n_time}")
print(f"  Frequencies: {n_freq}")
print(f"  Directions: {n_dir}")

print(f"\nReconstruction Method:")
print(f"  D(f,θ) = 1/(2π) × (1 + 2a₁cos(θ) + 2b₁sin(θ) + 2a₂cos(2θ) + 2b₂sin(2θ))")
print(f"  S_extended(f,θ) = S_f(f) × D(f,θ)")

print(f"\nError Summary:")
print(f"  Mean Relative L2 Error: {np.mean(relative_l2_error):.4f} ± {np.std(relative_l2_error):.4f}")
print(f"  Mean MSE: {np.mean(mse_per_time):.6e}")
print(f"  Mean MAE: {np.mean(mae_per_time):.6e}")

print(f"\nTop 5 Best Reconstruction (lowest error):")
best_idx = np.argsort(relative_l2_error)[:5]
for i, idx in enumerate(best_idx, 1):
    print(f"  {i}. Time step {idx}: L2 error = {relative_l2_error[idx]:.6f}")

print(f"\nTop 5 Worst Reconstruction (highest error):")
worst_idx = np.argsort(relative_l2_error)[-5:][::-1]
for i, idx in enumerate(worst_idx, 1):
    print(f"  {i}. Time step {idx}: L2 error = {relative_l2_error[idx]:.6f}")

print("\n" + "=" * 70)
print("SECOND HARMONIC COMPLETE")
print("=" * 70)
