#!/usr/bin/env python

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_to_output


def find_column_by_partial_match(columns, substring):
    """Helper to robustly find a column whose name contains a given substring."""
    matches = [c for c in columns if substring.lower() in c.lower()]
    if not matches:
        raise KeyError(f"No column containing '{substring}' found in columns: {list(columns)}")
    if len(matches) > 1:
        print(f"[WARN] Multiple columns match '{substring}'. Using '{matches[0]}'.")
    return matches[0]


def preprocess_subject_to_3d(
    subject_dir: Path,
    out_path: Path,
    ignore_zeros: bool = False,
):
    """
    Build a single 3D volume for nnUNet from per-file NIfTI images in `subject_dir`
    and save to `out_path`.

    Each file may be:
      - 2D:        (H, W)
      - small 3D:  (Z, H, W) or (H, W, Z)
      - complex 2-channel: (2, H, W) with [real, imag] -> magnitude

    We standardize everything to (H, W, Z_local) and then concatenate along Z to get
    a full 3D volume of shape (H, W, Z_total).

    Steps:
      1. Load all .nii/.nii.gz files in the subject directory (sorted).
      2. Convert complex (2,H,W) to magnitude (H,W).
      3. Stack into one 3D volume (H, W, Z_total).
      4. Compute global z-score across all voxels.
      5. Resample to isotropic [1, 1, 1] mm.
      6. Save as NIfTI .nii.gz at out_path.
    """

    nii_files = sorted([p for p in subject_dir.glob("*.nii*") if p.is_file()])
    if not nii_files:
        print(f"[WARN] No NIfTI files found in {subject_dir}")
        return

    print(f"Processing subject {subject_dir.name}: {len(nii_files)} files found")

    # ---- inspect first file to get base geometry ----
    ref_img = nib.load(str(nii_files[0]))
    ref_data = ref_img.get_fdata(dtype=np.float32)

    if ref_data.ndim == 2:
        # plain magnitude 2D slice
        H, W = ref_data.shape
        Z0 = 1
    elif ref_data.ndim == 3 and ref_data.shape[0] == 2:
        # SPECIAL CASE: complex data [real, imag] with shape (2, H, W)
        H, W = ref_data.shape[1], ref_data.shape[2]
        Z0 = 1
    elif ref_data.ndim == 3:
        # Could be (Z, H, W) or (H, W, Z).
        # Heuristic: if first dim is smaller than the others, treat as (Z, H, W).
        if ref_data.shape[0] <= ref_data.shape[1] and ref_data.shape[0] <= ref_data.shape[2]:
            # (Z, H, W) -> we'll transpose to (H, W, Z)
            Z0 = ref_data.shape[0]
            H, W = ref_data.shape[1], ref_data.shape[2]
        else:
            # Assume (H, W, Z)
            H, W = ref_data.shape[0], ref_data.shape[1]
            Z0 = ref_data.shape[2]
    else:
        raise ValueError(
            f"Unexpected data shape in {nii_files[0]}: {ref_data.shape}. "
            f"Expected 2D or 3D."
        )

    zooms = ref_img.header.get_zooms()
    if len(zooms) >= 3:
        dx, dy, dz = float(zooms[0]), float(zooms[1]), float(zooms[2])
    elif len(zooms) == 2:
        dx, dy = float(zooms[0]), float(zooms[1])
        dz = 1.0
    else:
        dx = dy = dz = 1.0

    # ---- load and stack everything into one 3D volume ----
    slices_list = []
    total_z = 0

    for f in nii_files:
        img = nib.load(str(f))
        data = img.get_fdata(dtype=np.float32)

        if data.ndim == 2:
            # (H, W) -> (H, W, 1)
            if data.shape != (H, W):
                raise ValueError(
                    f"Inconsistent shape in {f}: got {data.shape}, expected {(H, W)}"
                )
            vol_local = data[..., None]  # (H, W, 1)

        elif data.ndim == 3 and data.shape[0] == 2 and data.shape[1:] == (H, W):
            # SPECIAL CASE: complex [real, imag] -> magnitude (H, W, 1)
            real = data[0, :, :]
            imag = data[1, :, :]
            complex_img = real + 1j * imag
            mag = np.abs(complex_img).astype(np.float32)
            if mag.shape != (H, W):
                raise ValueError(
                    f"Complex magnitude shape mismatch in {f}: got {mag.shape}, expected {(H, W)}"
                )
            vol_local = mag[..., None]  # (H, W, 1)

        elif data.ndim == 3:
            # Try to normalize to (H, W, Z_local)
            if data.shape == (H, W, data.shape[2]):
                # Already (H, W, Z_local)
                vol_local = data
            elif data.shape[0] <= data.shape[1] and data.shape[0] <= data.shape[2]:
                # Likely (Z, H, W) -> transpose to (H, W, Z)
                if data.shape[1] != H or data.shape[2] != W:
                    raise ValueError(
                        f"Inconsistent 3D shape in {f}: got {data.shape}, "
                        f"expected (?, {H}, {W})"
                    )
                vol_local = np.transpose(data, (1, 2, 0))  # (H, W, Z_local)
            elif data.shape[2] <= data.shape[0] and data.shape[2] <= data.shape[1]:
                # Possibly (H, W, Z_local) with small Z - accept as is
                if data.shape[0] != H or data.shape[1] != W:
                    raise ValueError(
                        f"Inconsistent 3D shape in {f}: got {data.shape}, "
                        f"expected ({H}, {W}, ?)"
                    )
                vol_local = data
            else:
                raise ValueError(
                    f"Ambiguous 3D shape in {f}: {data.shape}. "
                    "Cannot determine which axis is Z."
                )
        else:
            raise ValueError(
                f"Unexpected data dimension in {f}: {data.ndim}. Only 2D or 3D supported."
            )

        slices_list.append(vol_local)
        total_z += vol_local.shape[2]

    # Concatenate along Z
    vol3d = np.concatenate(slices_list, axis=2)  # (H, W, Z_total)
    print(f"  Assembled volume shape (before z-score): {vol3d.shape}")

    # ---- global z-score normalization ----
    if ignore_zeros:
        mask = np.isfinite(vol3d) & (vol3d != 0)
    else:
        mask = np.isfinite(vol3d)

    if not np.any(mask):
        raise RuntimeError(f"No valid voxels found in {subject_dir} to compute mean/std.")

    vals = vol3d[mask]
    mean = float(vals.mean())
    std = float(vals.std() + 1e-8)

    print(f"  Global mean: {mean:.4f}, std: {std:.4f}")

    vol3d_norm = (vol3d - mean) / std

    # ---- construct NIfTI + resample to [1,1,1] ----
    affine = np.diag([dx, dy, dz, 1.0])
    vol_img = nib.Nifti1Image(vol3d_norm.astype(np.float32), affine)
    resampled_img = resample_to_output(vol_img, voxel_sizes=(1.0, 1.0, 1.0), order=3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(resampled_img, str(out_path))
    print(f"  Saved 3D preprocessed volume to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess MRI into 3D volumes for nnUNet: "
            "complex->magnitude, z-score, 1mm iso, one NIfTI per malignant subject."
        )
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default="/ess/scratch/scratch1/rachelgordon/zf_data_192_slices",
        help="Root directory containing fastMRI_breast_*_2 subject folders.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/ess/scratch/scratch1/rachelgordon/zf_data_192_slices/preprocessed_for_seg",
        help="Directory where preprocessed 3D NIfTI volumes will be saved.",
    )
    parser.add_argument(
        "--metadata_xlsx",
        type=str,
        default=(
            "/gpfs/data/karczmar-lab/workspaces/rachelgordon/"
            "breastMRI-recon/ddei/data/breast_fastMRI_final.xlsx"
        ),
        help="Path to Excel file with lesion status metadata.",
    )
    parser.add_argument(
        "--ignore_zeros",
        action="store_true",
        help="Ignore zero-valued voxels when computing global mean/std.",
    )

    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    # -------------------------
    # Load metadata and filter to malignant cases
    # -------------------------
    print(f"Loading metadata from: {args.metadata_xlsx}")
    df = pd.read_excel(args.metadata_xlsx)

    patient_col = find_column_by_partial_match(df.columns, "Patient Coded Name")
    lesion_col = find_column_by_partial_match(df.columns, "Lesion status")

    malignant_df = df[df[lesion_col] == 1]
    malignant_ids = set(malignant_df[patient_col].astype(str))

    print(f"Found {len(malignant_ids)} malignant patients in metadata.")

    if not malignant_ids:
        print("[ERROR] No malignant cases (lesion status = 1) found. Exiting.")
        return

    subjects_to_process = []
    for pid in sorted(malignant_ids):
        subj_dir_name = f"{pid}_2"  # e.g., fastMRI_breast_010 -> fastMRI_breast_010_2
        subj_dir = input_root / subj_dir_name
        if subj_dir.is_dir():
            subjects_to_process.append(subj_dir)
        else:
            print(f"[WARN] Malignant patient {pid} has no directory {subj_dir} under {input_root}")

    if not subjects_to_process:
        print("[ERROR] No subject directories found for malignant cases. Check paths/naming.")
        return

    print(f"Will preprocess {len(subjects_to_process)} malignant subject directories.")

    for subj_dir in subjects_to_process:
        out_fname = subj_dir.name + "_0000.nii.gz"  # e.g., fastMRI_breast_010_2_0000.nii.gz
        out_path = output_root / out_fname

        preprocess_subject_to_3d(
            subject_dir=subj_dir,
            out_path=out_path,
            ignore_zeros=args.ignore_zeros,
        )


if __name__ == "__main__":
    main()

