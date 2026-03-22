import numpy as np
import os
from scipy.optimize import fsolve
import warnings

# ============================================================================
# 1️⃣ LOAD PROCESSED DATA
# ============================================================================

print("=" * 70)
print("STEP 1: Loading processed data")
print("=" * 70)

processed_file = os.path.join("data", "processed", "era5_wave_training.npz")

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
# 2️⃣ EXTRACT INPUTS (ONLY S_f, a1, b1 - NO a2, b2)
# ============================================================================

print("\n" + "=" * 70)
print("STEP 2: Extracting inputs S_f, a1, b1 (first harmonic only)")
print("=" * 70)

S_f = X[:, 0, :]  # Frequency spectrum (time, 30)
a1 = X[:, 1, :]   # Directional moment a1: (time, 30)
b1 = X[:, 2, :]   # Directional moment b1: (time, 30)
# NOTE: Ignoring a2 = X[:, 3, :] and b2 = X[:, 4, :]

print(f"S_f shape: {S_f.shape}")
print(f"a1 shape: {a1.shape}")
print(f"b1 shape: {b1.shape}")

print(f"\nS_f range: [{np.nanmin(S_f):.6e}, {np.nanmax(S_f):.6e}]")
print(f"a1 range: [{np.nanmin(a1):.4f}, {np.nanmax(a1):.4f}]")
print(f"b1 range: [{np.nanmin(b1):.4f}, {np.nanmax(b1):.4f}]")

# ============================================================================
# 3️⃣ MEM DIRECTIONAL DISTRIBUTION FUNCTIONS (FIRST HARMONIC ONLY)
# ============================================================================

def compute_mem_distribution(theta, lambdas):
    """
    Compute MEM directional distribution D(θ) = exp(F(θ)) / Z
    where F(θ) = λ₁cos θ + λ₂sin θ (first harmonic only)
    """
    l1, l2 = lambdas
    
    # F(θ) = λ₁cos θ + λ₂sin θ (no second harmonic terms)
    F = l1 * np.cos(theta) + l2 * np.sin(theta)
    
    # Prevent overflow by subtracting maximum
    F_max = np.max(F)
    exp_F = np.exp(F - F_max)
    
    # Normalize: D(θ) = exp(F(θ)) / ∫ exp(F(θ)) dθ
    ddir = theta[1] - theta[0] if len(theta) > 1 else 2*np.pi/len(theta)
    Z = np.sum(exp_F) * ddir
    
    return exp_F / Z

def compute_moments_from_distribution(theta, D, ddir):
    """Compute directional moments from distribution D(θ) (first harmonic only)"""
    a1_calc = np.sum(D * np.cos(theta)) * ddir
    b1_calc = np.sum(D * np.sin(theta)) * ddir  
    return np.array([a1_calc, b1_calc])

def moment_constraints(lambdas, theta, target_moments, ddir):
    """
    Constraint equations: computed moments - target moments = 0
    (first harmonic only: a1, b1)
    """
    D = compute_mem_distribution(theta, lambdas)
    computed_moments = compute_moments_from_distribution(theta, D, ddir)
    return computed_moments - target_moments

def solve_mem_parameters(a1_val, b1_val, theta, ddir, max_iter=100):
    """
    Solve for MEM parameters λ₁, λ₂ given target moments a1, b1
    (first harmonic only)
    """
    target_moments = np.array([a1_val, b1_val])
    
    # Handle zero/small moments case
    if np.max(np.abs(target_moments)) < 1e-10:
        return np.zeros(2), True  # Uniform distribution case
    
    # Initial guess: small random values
    initial_guess = np.random.normal(0, 0.1, 2)
    
    try:
        # Solve the nonlinear system
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            solution = fsolve(
                moment_constraints, 
                initial_guess, 
                args=(theta, target_moments, ddir),
                maxfev=max_iter * 4
            )
        
        # Verify solution quality
        residual = moment_constraints(solution, theta, target_moments, ddir)
        max_error = np.max(np.abs(residual))
        
        success = max_error < 1e-4
        return solution, success
        
    except Exception:
        return np.zeros(2), False

# ============================================================================
# 4️⃣ RECONSTRUCT MEM SPECTRUM (FIRST HARMONIC ONLY)
# ============================================================================

print("\n" + "=" * 70)
print("STEP 3: Reconstructing spectrum using MEM (first harmonic only)")
print("=" * 70)

# Convert directions to radians
direction_rad = np.deg2rad(direction_deg)
ddir = np.deg2rad(15.0)  # ERA5 uses 15° spacing

n_time = X.shape[0]
n_freq = 30
n_dir = 24

S_mem = np.zeros((n_time, n_freq, n_dir))

# Track solution statistics
total_cases = n_time * n_freq
solved_cases = 0
failed_cases = 0

print(f"Solving MEM parameters for {total_cases} cases...")

for t in range(n_time):
    if t % 100 == 0:
        print(f"  Processing time step {t}/{n_time}")
    
    for f in range(n_freq):
        S_f_val = S_f[t, f]
        a1_val = a1[t, f]
        b1_val = b1[t, f]
        
        if S_f_val > 1e-15:  # Only process non-zero frequencies
            # Solve for MEM parameters (only λ₁, λ₂)
            lambdas, success = solve_mem_parameters(
                a1_val, b1_val, 
                direction_rad, ddir
            )
            
            if success:
                # Compute MEM distribution
                D = compute_mem_distribution(direction_rad, lambdas)
                solved_cases += 1
            else:
                # Fallback to uniform distribution
                D = np.ones(n_dir) / n_dir
                failed_cases += 1
                
        else:
            # No energy case - uniform distribution
            D = np.ones(n_dir) / n_dir
            
        # S_mem(f, θ) = S_f(f) × D(f, θ)
        S_mem[t, f, :] = S_f_val * D

print(f"\n✓ MEM reconstruction completed")
print(f"  Successfully solved: {solved_cases}/{total_cases} ({100*solved_cases/total_cases:.1f}%)")
print(f"  Failed cases: {failed_cases}/{total_cases} ({100*failed_cases/total_cases:.1f}%)")
print(f"  S_mem shape: {S_mem.shape} (time, 30, 24)")
print(f"  S_mem range: [{np.nanmin(S_mem):.6e}, {np.nanmax(S_mem):.6e}]")

# ============================================================================
# 5️⃣ COMPUTE ERROR
# ============================================================================

print("\n" + "=" * 70)
print("STEP 4: Computing reconstruction error") 
print("=" * 70)

# Remove NaN values for error computation
y_true_clean = np.nan_to_num(y_true, nan=0.0)
S_mem_clean = np.nan_to_num(S_mem, nan=0.0)

# Relative L2 error per time step
epsilon = 1e-10  # avoid division by zero

norm_true = np.linalg.norm(y_true_clean.reshape(n_time, -1), axis=1)
norm_diff = np.linalg.norm((S_mem_clean - y_true_clean).reshape(n_time, -1), axis=1)

relative_l2_error = norm_diff / (norm_true + epsilon)

# MSE per time step
mse_per_time = np.mean((S_mem_clean - y_true_clean) ** 2, axis=(1, 2))

# Mean Absolute Error per time step
mae_per_time = np.mean(np.abs(S_mem_clean - y_true_clean), axis=(1, 2))

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
# 6️⃣ PRINT RESULTS
# ============================================================================

print("\n" + "=" * 70)
print("MEM RECONSTRUCTION RESULTS (FIRST HARMONIC ONLY)")
print("=" * 70)

print(f"\nOverall Statistics:")
print(f"  Total time steps: {n_time}")
print(f"  Frequencies: {n_freq}")
print(f"  Directions: {n_dir}")
print(f"  MEM solution success rate: {100*solved_cases/total_cases:.1f}%")

print(f"\nReconstruction Method:")
print(f"  D(f,θ) = exp(λ₁cos θ + λ₂sin θ) / Z (first harmonic only)")
print(f"  Z = ∫ exp(λ₁cos θ + λ₂sin θ) dθ")
print(f"  Moment constraints:")
print(f"    a₁ = ∫ D(θ)cos θ dθ")
print(f"    b₁ = ∫ D(θ)sin θ dθ") 
print(f"  S_mem(f,θ) = S_f(f) × D(f,θ)")

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
print("MEM COMPLETE (FIRST HARMONIC ONLY)")
print("=" * 70)
