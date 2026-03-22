import os
import xarray as xr
import numpy as np
import sys
DATA_ROOT = "eval_data/gum/2019/01/"  # Change this to "train_data/2017" for training data
OUTPUT_DIR = os.path.join(DATA_ROOT, "processed_1")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "era5_wave_eval.npz")


def construct_frequency_direction():
    # f₁ = 0.03453 Hz
    # f(n) = f(n-1) × 1.1
    f1 = 0.03453
    ratio = 1.1
    freq_hz = np.array([f1 * (ratio ** (n - 1)) for n in range(1, 31)])

    # 24 directions, starting at 7.5°, incremented by 15°
    direction_deg = np.array([7.5 + 15 * n for n in range(24)])
    direction_rad = np.deg2rad(direction_deg)
    ddir = np.deg2rad(15.0)

    return freq_hz, direction_deg, direction_rad, ddir


def reorder_to_time_frequency_direction(d2fd):
    d2fd = np.squeeze(d2fd)

    if d2fd.ndim != 3:
        raise ValueError(f"Unexpected d2fd dimensions after squeeze: {d2fd.shape}")

    # Target: (time, frequency, direction)
    if d2fd.shape[1] == 24 and d2fd.shape[2] == 30:
        d2fd = np.swapaxes(d2fd, 1, 2)
    elif d2fd.shape[1] == 30 and d2fd.shape[2] == 24:
        pass
    else:
        raise ValueError(f"Unexpected d2fd shape: {d2fd.shape}")

    return d2fd


def process_file(nc_path, direction_rad, ddir):
    ds = xr.open_dataset(nc_path)
    try:
        d2fd = ds["d2fd"].values
        
        # Handle spatial dimensions for evaluation data
        if 'eval' in DATA_ROOT:
            # For evaluation data with spatial grid (3x3), select center point
            if d2fd.ndim == 5:  # (time, direction, frequency, lat, lon)
                # Select center point of the 3x3 grid
                center_lat_idx = 1  # Middle of 3 points (0, 1, 2)
                center_lon_idx = 1  # Middle of 3 points (0, 1, 2)
                d2fd = d2fd[:, :, :, center_lat_idx, center_lon_idx]
        
        d2fd = reorder_to_time_frequency_direction(d2fd)

        # S = 10^(d2fd), units: m² · s · radian⁻¹
        S = 10.0 ** d2fd
        S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
        S[S < 1e-15] = 0.0

        # S_f: integrate over direction
        S_f = np.sum(S, axis=2) * ddir

        # a1, b1 first moments
        a1_integrated = np.sum(S * np.cos(direction_rad[None, None, :]), axis=2) * ddir
        b1_integrated = np.sum(S * np.sin(direction_rad[None, None, :]), axis=2) * ddir
        
        # a2, b2 second moments
        a2_integrated = np.sum(S * np.cos(2 * direction_rad[None, None, :]), axis=2) * ddir
        b2_integrated = np.sum(S * np.sin(2 * direction_rad[None, None, :]), axis=2) * ddir

        # For frequencies with very low energy, set directional moments to 0 (not NaN)
        # This is physically correct: no energy means no directional information
        energy_threshold = 1e-10
        has_energy = S_f > energy_threshold
        
        a1 = np.zeros_like(S_f)
        b1 = np.zeros_like(S_f)
        a2 = np.zeros_like(S_f)
        b2 = np.zeros_like(S_f)
        
        # Only compute directional moments where there is significant energy
        a1[has_energy] = a1_integrated[has_energy] / S_f[has_energy]
        b1[has_energy] = b1_integrated[has_energy] / S_f[has_energy]
        a2[has_energy] = a2_integrated[has_energy] / S_f[has_energy]
        b2[has_energy] = b2_integrated[has_energy] / S_f[has_energy]

        if "valid_time" in ds:
            valid_time = ds["valid_time"].values
        else:
            valid_time = np.arange(S.shape[0])

        return S, S_f, a1, b1, a2, b2, valid_time
    finally:
        ds.close()


def find_nc_files(root_dir):
    nc_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".nc"):
                nc_files.append(os.path.join(dirpath, filename))
    return sorted(nc_files)


def main():
    freq_hz, direction_deg, direction_rad, ddir = construct_frequency_direction()

    nc_files = find_nc_files(DATA_ROOT)
    if not nc_files:
        print(f"No .nc files found under {DATA_ROOT}")
        return
    
    # Limit to first 8 months of data (keep only 8 files)
    nc_files = nc_files#[:8]
    for f in nc_files:
        print(f"Found file: {f}")
   
    print(f"Processing {len(nc_files)} files (limited to 8 months)")

    S_list = []
    S_f_list = []
    a1_list = []
    b1_list = []
    a2_list = []
    b2_list = []
    time_list = []

    for nc_path in nc_files:
        print(f"Processing {nc_path} ...")
        try:
            S, S_f, a1, b1, a2, b2, valid_time = process_file(nc_path, direction_rad, ddir)
        except Exception as exc:
            print(f"  Skipping due to error: {exc}")
            continue

        S_list.append(S)
        S_f_list.append(S_f)
        a1_list.append(a1)
        b1_list.append(b1)
        a2_list.append(a2)
        b2_list.append(b2)
        time_list.append(valid_time)

        print(f"  S_f shape: {S_f.shape}")
        print(f"  a1 shape: {a1.shape}")
        print(f"  b1 shape: {b1.shape}")
        print(f"  a2 shape: {a2.shape}")
        print(f"  b2 shape: {b2.shape}")

    if not S_list:
        print("No files processed successfully.")
        return

    S_2d_all = np.concatenate(S_list, axis=0)
    S_f_all = np.concatenate(S_f_list, axis=0)
    a1_all = np.concatenate(a1_list, axis=0)
    b1_all = np.concatenate(b1_list, axis=0)
    a2_all = np.concatenate(a2_list, axis=0)
    b2_all = np.concatenate(b2_list, axis=0)
    valid_time_all = np.concatenate(time_list, axis=0)

    # Build training arrays
    # Input: S_f (30), a1 (30), b1 (30), a2 (30), b2 (30) -> shape (time, 5, 30)
    X = np.stack([S_f_all, a1_all, b1_all, a2_all, b2_all], axis=1)
    # Target: full S(f,θ) -> shape (time, 30, 24)
    y = S_2d_all

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez(
        OUTPUT_FILE,
        X=X,
        y=y,
        freq_hz=freq_hz,
        direction_deg=direction_deg,
        valid_time=valid_time_all,
    )

    print("\nProcessing complete")
    print(f"Total time steps: {X.shape[0]}")
    print(f"X shape: {X.shape} (time, 5, 30)")
    print(f"y shape: {y.shape} (time, 30, 24)")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
