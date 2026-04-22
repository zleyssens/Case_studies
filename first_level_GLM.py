"""
STEP 3 — First-level GLM (one model per participant)
=====================================================
Purpose: Fit a General Linear Model for each participant separately.
         This estimates how strongly each brain voxel responds to
         congruent vs incongruent trials (and vs baseline).

What happens here:
  - We load the preprocessed BOLD signal for each participant
  - We load the events file (which trial was congruent/incongruent and when)
  - We load motion/confound parameters from fMRIPrep
  - We fit a GLM using nilearn's FirstLevelModel
  - We compute three contrast maps per participant and save them as NIfTI files

Output:
  For each participant, three .nii.gz files are saved:
    - incongruent_minus_congruent.nii.gz  (the Stroop interference map)
    - task_minus_baseline.nii.gz          (overall task activation)
    - incongruent_minus_baseline.nii.gz   (incongruent trials only)
"""

import os
import glob
import numpy as np
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel
from nilearn import image
import nibabel as nib

# ─────────────────────────────────────────────
# PATHS — adjust these to match your setup
# ─────────────────────────────────────────────
BIDS_ROOT    = r"C:\Users\zoele\Desktop\case_studies_project\PIOP1_AOMIC"
FMRIPREP_DIR = os.path.join(BIDS_ROOT, "derivatives", "fmriprep")
OUTPUT_DIR   = r"C:\Users\zoele\Desktop\case_studies_project\first_level_GLM_contrasts"   # where contrast maps will be saved
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# WHICH CONFOUNDS TO INCLUDE
# ─────────────────────────────────────────────
# These are standard confound regressors from fMRIPrep:
#   - 6 motion parameters (translation + rotation in x/y/z)
#   - Their temporal derivatives (how fast motion changed)
#   - aCompCor components (account for physiological noise from CSF/WM)
#   - framewise_displacement is NOT used as a regressor (only for QC)
CONFOUND_COLS = [
    "trans_x", "trans_y", "trans_z",
    "rot_x",   "rot_y",   "rot_z",
    "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
    "rot_x_derivative1",   "rot_y_derivative1",   "rot_z_derivative1",
    "a_comp_cor_00", "a_comp_cor_01", "a_comp_cor_02",
    "a_comp_cor_03", "a_comp_cor_04",  # top 5 aCompCor components
]

# ─────────────────────────────────────────────
# FUNCTION: Load and clean confounds for one participant
# ─────────────────────────────────────────────
def load_confounds(confounds_path, columns):
    """
    Load selected confound columns from fMRIPrep's confounds TSV.
    The first row of fMRIPrep confounds is often NaN (no derivative at t=0).
    We fill those NaNs with 0 so the GLM doesn't crash.
    """
    df = pd.read_csv(confounds_path, sep="\t")

    # Only keep columns that actually exist in this file
    available = [c for c in columns if c in df.columns]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        print(f"  WARNING: These confounds not found and will be skipped: {missing}")

    df_conf = df[available].fillna(0)  # fill first-row NaNs with 0
    return df_conf


# ─────────────────────────────────────────────
# FUNCTION: Load and standardise events for one participant
# ─────────────────────────────────────────────
def load_events(events_path):
    """
    Load the events TSV and make sure trial_type labels are clean.
    PIOP1 gender-Stroop events should have columns:
      onset, duration, trial_type
    where trial_type is something like 'congruent' / 'incongruent'.

    We also add a catch-all 'other' condition for any trial types we
    don't care about (e.g. fixation, button presses) so the GLM
    accounts for them rather than treating them as baseline.
    """
    df = pd.read_csv(events_path, sep="\t")

    # Print available trial types the first time you run this
    # so you can verify the labels match what you expect
    print(f"    Trial types in events file: {df['trial_type'].unique()}")

    # Keep only the columns nilearn needs
    df = df[["onset", "duration", "trial_type"]].copy()

    return df


# ─────────────────────────────────────────────
# MAIN LOOP: Fit GLM for every participant
# ─────────────────────────────────────────────
# Find all preprocessed BOLD files for the gstroop task
bold_files = sorted(glob.glob(
    os.path.join(FMRIPREP_DIR, "sub-*", "func",
                 "*task-gstroop*space-MNI152NLin2009cAsym*_desc-preproc_bold.nii.gz")))

print(f"Found {len(bold_files)} BOLD files to process.\n")

import gzip

path = r"C:\Users\zoele\Desktop\case_studies_project\PIOP1_AOMIC\derivatives\fmriprep\sub-0001\func\sub-0001_task-gstroop_acq-seq_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"

with open(path, "rb") as f:
    print(f.read(2))

# Track which participants succeed or fail
success_list = []
failed_list  = []

for bold_path in bold_files:

    # ── Extract subject ID from the filename ──────────────────────────────
    # Filename looks like: sub-0001_task-gstroop_..._preproc_bold.nii.gz
    fname   = os.path.basename(bold_path)
    sub_id  = fname.split("_")[0]   # e.g. 'sub-0001'
    print(f"Processing {sub_id}...")

    # ── Build output folder for this subject ─────────────────────────────
    sub_out = os.path.join(OUTPUT_DIR, sub_id)
    os.makedirs(sub_out, exist_ok=True)

    # ── Skip if already done (useful if the loop crashes halfway) ─────────
    done_flag = os.path.join(sub_out, "incongruent_minus_congruent.nii.gz")
    if os.path.exists(done_flag):
        print(f"  Already processed, skipping.")
        success_list.append(sub_id)
        continue

    try:
        # ── Find the matching events and confounds files ──────────────────
        # Events are in the raw BIDS folder, confounds in fmriprep derivatives
        events_path = os.path.join(
            BIDS_ROOT, sub_id, "func",
            f"{sub_id}_task-gstroop_acq-seq_events.tsv"
        )
        confounds_path = os.path.join(
            FMRIPREP_DIR, sub_id, "func",
            f"{sub_id}_task-gstroop_acq-seq_desc-confounds_regressors.tsv"
        )

        # Verify files exist before going further
        if not os.path.exists(events_path):
            raise FileNotFoundError(f"Events file not found: {events_path}")
        if not os.path.exists(confounds_path):
            raise FileNotFoundError(f"Confounds file not found: {confounds_path}")

        # ── Load events and confounds ─────────────────────────────────────
        events    = load_events(events_path)
        confounds = load_confounds(confounds_path, CONFOUND_COLS)

        # ── Get the TR (repetition time) from the BOLD NIfTI header ───────
        # TR is the time between successive brain volumes (in seconds)
        img = nib.load(bold_path)
        t_r = img.header.get_zooms()[3]   # 4th zoom = time dimension
        print(f"  TR = {t_r}s | Volumes = {img.shape[3]} | "
              f"Confounds shape = {confounds.shape}")

        # ── Fit the First-Level GLM ───────────────────────────────────────
        # Key parameters explained:
        #   t_r           : repetition time of the fMRI scan
        #   noise_model   : AR(1) accounts for temporal autocorrelation
        #   standardize   : False — we want raw parameter estimates (betas)
        #   hrf_model     : 'spm' = the standard SPM HRF (double-gamma)
        #   drift_model   : cosine high-pass filter (equivalent to 128s cutoff)
        #   high_pass     : 1/128 Hz cutoff frequency
        #   signal_scaling: False — do not z-score the BOLD signal

        fmri_glm = FirstLevelModel(
            t_r=t_r,
            noise_model="ar1",
            standardize=False,
            hrf_model="spm",
            drift_model="cosine",
            high_pass=1.0 / 128,
            signal_scaling=False,
            verbose=0,
        )

        # Fit the model: takes the 4D BOLD image, event timing, and confounds
        fmri_glm.fit(bold_path, events=events, confounds=confounds)

        # ── Compute contrasts ─────────────────────────────────────────────
        # Each contrast is a weighted combination of the model's regressors.
        # nilearn accepts plain strings if the condition names match exactly.

        # CONTRAST 1: Incongruent − Congruent (the Stroop interference effect)
        # This is the key map for H1 and H3
        c1 = fmri_glm.compute_contrast(
            "incongruent - congruent",
            output_type="effect_size"    # returns beta (effect size) map
        )

        # CONTRAST 2: Task (all trials) − Baseline
        # 0.5 * each condition = average of both vs rest; used for H2
        c2 = fmri_glm.compute_contrast(
            "0.5 * incongruent + 0.5 * congruent",
            output_type="effect_size"
        )

        # CONTRAST 3: Incongruent − Baseline (useful for visualisation)
        c3 = fmri_glm.compute_contrast(
            "incongruent",
            output_type="effect_size"
        )

        # NOTE: If the condition names in your events file are different
        # (e.g. 'gender_incongruent', 'match', 'mismatch'), update the
        # strings above to match. Check the print output from load_events().

        # ── Save contrast maps ────────────────────────────────────────────
        nib.save(c1, os.path.join(sub_out, "incongruent_minus_congruent.nii.gz"))
        nib.save(c2, os.path.join(sub_out, "task_minus_baseline.nii.gz"))
        nib.save(c3, os.path.join(sub_out, "incongruent_minus_baseline.nii.gz"))

        print(f"  ✓ Done. Contrast maps saved to {sub_out}")
        success_list.append(sub_id)

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed_list.append((sub_id, str(e)))

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("\n" + "="*50)
print(f"Completed: {len(success_list)} participants")
print(f"Failed:    {len(failed_list)} participants")
if failed_list:
    print("Failed subjects:")
    for sub, reason in failed_list:
        print(f"  {sub}: {reason}")