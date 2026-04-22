path = r"C:\Users\zoele\Desktop\case_studies_project\PIOP1_AOMIC\derivatives\fmriprep\sub-0167\func\sub-0167_task-gstroop_acq-seq_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"

with open(path, "rb") as f:
    print(f.read(4))