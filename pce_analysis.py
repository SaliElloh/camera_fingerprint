import src.Functions as Fu
import src.Filter as Ft
import src.getFingerprint as gF
import src.maindir as md
import src.extraUtils as eu
import numpy as np
import os
import cv2 as cv
import pickle
import matplotlib.pyplot as plt
# import pandas as pd

camera_dir = '/data/Sali/camera_fingerprint/Vision_dataset_organized'
camera_lst = ['D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09', 'D10',]


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

def PRNU_calculator_for_multiple_cameras(cameras_dir, camera_lst):
    # extracting Fingerprint from same size images in a path

    """extracts PRNU fingerprints from multiple cameras and returns a list of tuples
    where each tuple contains the camera name and its corresponding fingerprint matrix.
    :param cameras_dir: Directory containing subdirectories for each camera
    :param camera_lst: List of camera subdirectory names
    :return: List of tuples (camera_name, fingerprint_matrix)
    """
    
    list_of_fingerprints = []
    for camera in camera_lst:
        camera_dir = os.path.join(cameras_dir, camera)

        flat_imgs_dir = os.path.join(camera_dir, 'flat')

        Images = [os.path.join(flat_imgs_dir, fname) for fname in os.listdir(flat_imgs_dir) if fname.endswith('.jpg')][:25]


        RP,_,_ = gF.getFingerprint(Images)
        RP = Fu.rgb2gray1(RP)
        sigmaRP = np.std(RP)
        Fingerprint = Fu.WienerInDFT(RP, sigmaRP)

        list_of_fingerprints.append((camera, Fingerprint))

######## FIND A WAY TO SAVE THE FINGERPRINTS INTO A DATAFRAME ###############
    # with open('fingerprints_list.pkl', 'wb') as file:
    #     pickle.dump(list_of_fingerprints, file)

    # data = np.array(list_of_fingerprints, dtype=object)
    # np.save('fingerprints_list.npy', data)

    return list_of_fingerprints

list_of_fingerprints = PRNU_calculator_for_multiple_cameras(camera_dir, camera_lst)

def PCE_array(camera_lst, PRNU_lst, test_img_path):

    """"extracts PCE values for a test image against multiple camera PRNU fingerprints
    :param camera_lst: List of camera IDs
    :param PRNU_lst: List of tuples containing camera names and their corresponding PRNU fingerprints
    :param test_img_path: Path to the test image
    :return: List of PCE values for each camera"""
    
    PCE_values = []

    for camera, camera_PRNU_pair in zip(camera_lst, PRNU_lst):
        PRNU = camera_PRNU_pair[1]

        det = PCE(PRNU, test_img_path)
        pce = det['PCE']
        PCE_values.append(pce)

    plt.figure(figsize=(10, 6))

    plt.plot(camera_lst, PCE_values, marker='o', linestyle='-', color='b')

    # Annotate peak
    max_index = PCE_values.index(max(PCE_values))
    plt.plot(camera_lst[max_index], PCE_values[max_index], 'ro')  # red dot on peak
    plt.text(camera_lst[max_index], PCE_values[max_index]*1.05,
             f"Peak: {PCE_values[max_index]:.2f}", ha='center', fontsize=10, color='red')

    plt.title('PCE Line Plot - Test Image vs Camera PRNUs')
    plt.xlabel('Camera ID')
    plt.ylabel('PCE Value')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return PCE_values

test_img_path = '/data/Sali/camera_fingerprint/Vision_dataset_organized/D01/nat/D01_I_nat_0002.jpg'

pce_list = PCE_array(camera_lst, list_of_fingerprints, test_img_path)

test_img_path = '/data/Sali/camera_fingerprint/Vision_dataset_organized/D07/flat/D07_I_flat_0002.jpg'

pce_list = PCE_array(camera_lst, list_of_fingerprints, test_img_path)

