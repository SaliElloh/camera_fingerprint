import numpy as np
import os
import cv2 as cv
import pickle
import pandas as pd
from sklearn.metrics import roc_curve
import src.Functions as Fu
import src.Filter as Ft
import src.getFingerprint 
import src.maindir as md
# import matplotlib.pyplot as plt
# import seaborn as sns

camera_dir = '/scratch/hafiz_root/hafiz1/selloh/Dresden_sample_dataset'
camera_lst = os.listdir(camera_dir)
pce_res_dir = '/home/selloh/camera_fingerprint/pce_results'
fingerprint_dir = '/home/selloh/camera_fingerprint/dresden_prnu'
os.makedirs(pce_res_dir, exist_ok=True)

all_img_paths = []
for cam in os.listdir(camera_dir):
    cam_dir = os.path.join(camera_dir, cam, 'test')
    for img in os.listdir(cam_dir)[:20]:
        img_path = os.path.join(cam_dir, img)
        all_img_paths.append(img_path)



def PCE(Fingerprint, test_img_path):

    """    :param Fingerprint: Fingerprint matrix extracted from a set of images
    :param test_img_path: Path to the test image
    :return: Dictionary with detection results
    """
   
    test_img = cv.imread(test_img_path)
    if Fingerprint.shape != test_img.shape:
        
        test_img = cv.resize(test_img, (Fingerprint.shape[1], Fingerprint.shape[0]))        

    Noisex = Ft.NoiseExtractFromImage(test_img, sigma=2.)
    Noisex = Fu.WienerInDFT(Noisex, np.std(Noisex))

    # The optimal detector (see publication "Large Scale Test of Sensor Fingerprint Camera Identification")
    Ix = cv.cvtColor(test_img,# image in BGR format
                    cv.COLOR_BGR2GRAY)

    C = Fu.crosscorr(Noisex,np.multiply(Ix, Fingerprint))

    det, det0 = md.PCE(C)

    return det

def PCE_array(fingerprint, test_img_path):

    det = PCE(fingerprint, test_img_path)
    pce = det['PCE']
    
    return pce

for fname in os.listdir(fingerprint_dir):

    if fname.endswith('.pkl'):

        fpath = os.path.join(fingerprint_dir, fname)
        
        try: 
            with open(fpath, 'rb') as f:
                data = pickle.load(f)

        except Exception as e:
            print(f'failed to process {fpath}: {e}')
            continue

        fingerprint_name = data['camera_name']
        if fingerprint_name not in camera_lst:
            continue            
        print(f"processing {fingerprint_name}")

        df = pd.DataFrame(columns=['camera_fingerprint', 'test_data', 'test_img', 'pce_value'])

        for img in all_img_paths:
            img_array = img.split('/')
            test_dataset = img_array[5]
            test_img = img_array[6]

            pce = PCE_array(data['fingerprint_matrix'], img)
            df.loc[len(df)] = [fingerprint_name, test_dataset, test_img, pce]

        save_path = os.path.join(pce_res_dir, f"{fingerprint_name}.csv")
        df.to_csv(save_path, index=False)

print(f"finished processing all camera fingerprints")

        

