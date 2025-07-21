import numpy as np
import os
import cv2 as cv
import pickle
import pandas as pd

# import src.Filter as Ft
import src.Functions as Fu
import src.getFingerprint as gF
# import src.maindir as md
# import src.extraUtils as eu

camera_dir = '/scratch/hafiz_root/hafiz1/selloh/Dresden_sample_dataset'
camera_lst = os.listdir(camera_dir)
print(camera_lst)

prnu_output_dir = '/home/selloh/camera_fingerprint/dresden_prnu'

def PRNU_calculator_for_multiple_cameras(cameras_dir, camera_lst, output_dir):
    """
    Extracts PRNU fingerprints from multiple cameras and returns a DataFrame
    with columns: 'camera_name' and 'fingerprint_matrix'.
    Also saves the results in pickle, .npy, and .csv formats.
    """
    os.makedirs(output_dir, exist_ok=True)

    list_of_fingerprints = []

    for camera in camera_lst:
        camera_dir = os.path.join(cameras_dir, camera)
        flat_imgs_dir = os.path.join(camera_dir, 'flat')
        print(f' the flat images dir is {flat_imgs_dir}')

        Images = [os.path.join(flat_imgs_dir, fname)
                  for fname in os.listdir(flat_imgs_dir)
                  if fname.endswith('.JPG')][:100]
        RP, _, _ = gF.getFingerprint(Images)
        RP = Fu.rgb2gray1(RP)
        sigmaRP = np.std(RP)
        Fingerprint = Fu.WienerInDFT(RP, sigmaRP)

        save_path = os.path.join(output_dir, f"{camera}_fingerprint.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump({'camera_name': camera, 'fingerprint_matrix': Fingerprint}, f)

        list_of_fingerprints.append((camera, Fingerprint))

    df = pd.DataFrame(list_of_fingerprints, columns=['camera_name', 'fingerprint_matrix'])

    return df
PRNU_calculator_for_multiple_cameras(camera_dir, camera_lst, prnu_output_dir)

