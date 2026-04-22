"""
STEP 2 — Explore the PIOP1 dataset structure
=============================================
Purpose: Understand what files are available before running any analysis.
PIOP1 follows the BIDS format, so files are organized in a predictable way.

Run this script first to confirm your paths and check what data you have.
"""

import os
import glob
import pandas as pd

# ─────────────────────────────────────────────
# SET THIS to the root folder of your PIOP1 data
# ─────────────────────────────────────────────
BIDS_ROOT = r"C:\Users\zoele\Desktop\case_studies_project\PIOP1_AOMIC"  # <-- change this to your actual path

# ─────────────────────────────────────────────
# 1. List all subjects
# ─────────────────────────────────────────────
# In BIDS, each subject has a folder named sub-XXXX
subject_dirs = sorted(glob.glob(os.path.join(BIDS_ROOT, "sub-*")))
subject_ids = [os.path.basename(d) for d in subject_dirs if os.path.isdir(d)]

print(f"Total subjects found: {len(subject_ids)}")
print(f"First 5 subjects: {subject_ids[:5]}")
print(f"Last 5 subjects:  {subject_ids[-5:]}")

# ─────────────────────────────────────────────
# 2. Check one subject's folder structure
# ─────────────────────────────────────────────
# This tells you whether fMRI data, events, and confounds are where you expect
example_sub = subject_ids[0]
example_path = os.path.join(BIDS_ROOT, example_sub)

print(f"\nFolder structure for {example_sub}:")
for root, dirs, files in os.walk(example_path):
    level = root.replace(example_path, "").count(os.sep)
    indent = "  " * level
    print(f"{indent}{os.path.basename(root)}/")
    sub_indent = "  " * (level + 1)
    for f in files:
        print(f"{sub_indent}{f}")

# ─────────────────────────────────────────────
# 3. Find the preprocessed BOLD files for the gender-Stroop task
# ─────────────────────────────────────────────
# fMRIPrep outputs are typically inside derivatives/fmriprep/
# The gender-Stroop task is named 'gstroop' in PIOP1
FMRIPREP_DIR = os.path.join(BIDS_ROOT, "derivatives", "fmriprep")

bold_files = sorted(glob.glob(
    os.path.join(FMRIPREP_DIR, "sub-*", "func",
                 "*task-gstroop*space-MNI152NLin2009cAsym*preproc_bold.nii.gz")
))

print(f"\nPreprocessed BOLD files found: {len(bold_files)}")
for f in bold_files[:3]:
    print(f"  {os.path.basename(f)}")

# ─────────────────────────────────────────────
# 4. Find the events files (trial-level info)
# ─────────────────────────────────────────────
# Events files are in the raw BIDS folder (not derivatives)
# They tell us which trials were congruent vs incongruent
events_files = sorted(glob.glob(
    os.path.join(BIDS_ROOT, "sub-*", "func", "*task-gstroop*events.tsv")
))

print(f"\nEvents files found: {len(events_files)}")
if events_files:
    # Preview the first events file to understand trial structure
    print("\nPreview of first events file:")
    df = pd.read_csv(events_files[0], sep="\t")
    print(df.head(10).to_string())
    print(f"\nColumns: {list(df.columns)}")
    print(f"Unique trial types: {df['trial_type'].unique() if 'trial_type' in df.columns else 'column name may differ'}")

# ─────────────────────────────────────────────
# 5. Find the confounds files (motion parameters, etc.)
# ─────────────────────────────────────────────
# fMRIPrep outputs a confounds TSV per run with motion, aCompCor, etc.
confounds_files = sorted(glob.glob(
    os.path.join(FMRIPREP_DIR, "sub-*", "func",
                 "*task-gstroop*confounds_timeseries.tsv")
))

print(f"\nConfounds files found: {len(confounds_files)}")
if confounds_files:
    conf_df = pd.read_csv(confounds_files[0], sep="\t")
    print(f"Confounds available (first file): {list(conf_df.columns[:15])} ...")

# ─────────────────────────────────────────────
# 6. Load participant demographics (sex, age)
# ─────────────────────────────────────────────
# BIDS stores this in participants.tsv at the root level
participants_file = os.path.join(BIDS_ROOT, "participants.tsv")

if os.path.exists(participants_file):
    participants = pd.read_csv(participants_file, sep="\t")
    print(f"\nParticipants table shape: {participants.shape}")
    print(participants.head())
    print(f"\nSex distribution:\n{participants['sex'].value_counts()}")
    print(f"\nAge stats:\n{participants['age'].describe()}")
else:
    print(f"\nWARNING: participants.tsv not found at {participants_file}")
    print("Check if it exists under a different path or name.")