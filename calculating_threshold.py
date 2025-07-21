import numpy as np
import os
import cv2 as cv
import pickle
import pandas as pd
from sklearn.metrics import roc_curve
import src.Functions as Fu
import src.Filter as Ft
import src.getFingerprint as gF
import src.maindir as md
import matplotlib.pyplot as plt

# import altair as alt

camera_dir = '/scratch/hafiz_root/hafiz1/selloh/Dresden_sample_dataset'
test_dir = '/data/Sali/camera_fingerprint/test_dataset'
camera_lst = os.listdir(camera_dir)
pce_res_dir = '/home/selloh/camera_fingerprint/pce_results'
fingerprint_dir = '/home/selloh/camera_fingerprint/dresden_prnu'

# import numpy as np
import os
import cv2 as cv
import pickle
import pandas as pd
from sklearn.metrics import roc_curve
import src.Functions as Fu
import src.Filter as Ft
import src.getFingerprint as gF
import src.maindir as md
import plotly.graph_objects as go

import matplotlib.pyplot as plt

camera_dir = '/scratch/hafiz_root/hafiz1/selloh/Dresden_sample_dataset'
test_dir = '/data/Sali/camera_fingerprint/test_dataset'
pce_res_dir = '/home/selloh/camera_fingerprint/pce_results'
fingerprint_dir = '/home/selloh/camera_fingerprint/dresden_prnu'


list_of_dirs = [os.path.join(pce_res_dir, csv) for csv in os.listdir(pce_res_dir)]

for dir in list_of_dirs:
    df = pd.read_csv(dir)
    df['label'] = (df['camera_fingerprint'] == df['test_img']).astype(int)

    # True matches
    true_df = df[df["label"] == 1]
    false_df = df[df["label"] == 0]

    # Confirm labels are only 0 or 1
    assert set(df["label"].unique()) <= {0, 1}, "Labels must be 0 or 1 only."

    y_true = df["label"].values
    y_scores = df["pce_value"].values

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid()

    # Optional: Plot best threshold (Youden’s J)
    youden_idx = (tpr - fpr).argmax()
    optimal_thresh = thresholds[youden_idx]
    plt.axvline(fpr[youden_idx], linestyle='--', color='red', label=f"Best Threshold = {optimal_thresh:.2f}")
    plt.legend()
    plt.show()
