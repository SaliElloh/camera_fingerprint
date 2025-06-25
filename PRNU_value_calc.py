import shutil
import pywt
from glob import glob
from tqdm import tqdm
import os
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import train_test_split
import bm3d
from skimage import io, img_as_float
import matplotlib.pyplot as plt
from PIL import Image

# for denoising:
from skimage.restoration import denoise_wavelet, estimate_sigma
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio



# --- Configuration --- #
WAVELET = 'db8'
LEVEL = 4

# --- Wavelet Denoising for One Channel --- #
def wavelet_denoise(img_channel, wavelet=WAVELET, level=LEVEL):
    coeffs = pywt.wavedec2(img_channel, wavelet=wavelet, level=level)
    cA, cD = coeffs[0], coeffs[1:]

    # Remove high frequency details
    cD_filtered = []
    for detail in cD:
        cH, cV, cD = detail
        cH.fill(0)
        cV.fill(0)
        cD.fill(0)
        cD_filtered.append((cH, cV, cD))

    coeffs_filtered = [cA] + cD_filtered
    denoised = pywt.waverec2(coeffs_filtered, wavelet)
    return np.clip(denoised, 0, 255)

# --- Denoise and Compute Residual for RGB --- #
def process_rgb_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = img.astype(np.float32)
    B, G, R = cv2.split(img)

    R_denoised = wavelet_denoise(R)
    G_denoised = wavelet_denoise(G)
    B_denoised = wavelet_denoise(B)

    R_residual = R - R_denoised
    G_residual = G - G_denoised
    B_residual = B - B_denoised

    return (R, R_denoised, R_residual), (G, G_denoised, G_residual), (B, B_denoised, B_residual)


def flat_neighbors_equal(channel):
    Xh = channel - np.roll(channel, 1, axis=1)
    Xv = channel - np.roll(channel, 1, axis=0)
    Satur = np.logical_and(np.logical_and(Xh, Xv), 
                np.logical_and(np.roll(Xh, -1, axis=1),np.roll(Xv, -1, axis=0)))
    return Satur


# --- Run on one RGB image --- #
image_path = "/home/kavank/D01/flat/D01_I_flat_0001.jpg"  # <- Replace this path
R_data, G_data, B_data = process_rgb_image(image_path)

# --- Visualize Results --- #
def show_channel_results(channel_data, channel_name):
    original, denoised, residual = channel_data
    plt.figure(figsize=(15, 3))
    plt.suptitle(f'{channel_name} Channel', fontsize=14)

    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title('Original')

    plt.subplot(1, 3, 2)
    plt.imshow(denoised)
    plt.title('Denoised')

    plt.subplot(1, 3, 3)
    plt.imshow(residual)
    plt.title('Residual')

    plt.tight_layout()
    plt.show()

show_channel_results(R_data, 'Red')
show_channel_results(G_data, 'Green')
show_channel_results(B_data, 'Blue')



images_dir = '/home/kavank/D01/flat'
image_paths = glob(os.path.join(images_dir, '*.jpg'))

for path in image_paths:
    try:
        with Image.open(path) as img:
            img.verify()  # checks for corruption
    except Exception as e:
        print(f" Corrupted: {os.path.basename(path)} — {e}")


WAVELET = 'db8'
LEVEL = 4

# --- Wavelet Denoising for One Channel --- #
def wavelet_denoise(img_channel, wavelet=WAVELET, level=LEVEL):
    coeffs = pywt.wavedec2(img_channel, wavelet=wavelet, level=level)
    cA, cD = coeffs[0], coeffs[1:]

    # Remove high frequency details
    cD_filtered = []
    for detail in cD:
        cH, cV, cDD= detail
        cH.fill(0)
        cV.fill(0)
        cDD.fill(0)
        cD_filtered.append((cH, cV, cD))

    coeffs_filtered = [cA] + cD_filtered
    denoised = pywt.waverec2(coeffs_filtered, wavelet)
    return np.clip(denoised, 0, 255)

# --- Read Image Safely with PIL --- #
def read_image_pil(path):
    try:
        with Image.open(path) as img:
            img = img.convert('RGB')  # ensures 3 channels
            img_np = np.array(img).astype(np.float32)
            return img_np
    except Exception as e:
        print(f" Error reading image {path} with PIL: {e}")
        return None


# --- Denoise and Compute Residual for RGB --- #
def compute_residual(img, R,G,B):
    # change function's name
    # change the input to RGB channels instead of image path
    
    if img is None:
        return None, None, None

    R_denoised = wavelet_denoise(R)
    G_denoised = wavelet_denoise(G)
    B_denoised = wavelet_denoise(B)

    R_residual = R - R_denoised
    G_residual = G - G_denoised
    B_residual = B - B_denoised


    return R_residual, G_residual, B_residual


def prnu_calculator(W_list, I_list):
    """
    Compute MLE of PRNU (Equation 6)
    W_list: list of noise residuals (numpy arrays)
    I_list: list of corresponding intensity images (numpy arrays)
    Returns: PRNU fingerprint estimate (same shape as input images)
    """
    numerator = np.zeros_like(W_list[0])
    denominator = np.zeros_like(I_list[0])

    for Wk, Ik in zip(W_list, I_list):
        numerator += Wk * Ik
        denominator += Ik * Ik

    K_hat = numerator / denominator
    return K_hat

def rgb_to_grayscale(K_R: np.ndarray, K_G: np.ndarray, K_B: np.ndarray) -> np.ndarray:
    """
    Converts RGB channels to grayscale using the weighted average formula.
    
    Equation (8): K̂ = 0.3K̂_R + 0.6K̂_G + 0.1K̂_B

    """
    # Apply the weighted average formula: K̂ = 0.3K̂_R + 0.6K̂_G + 0.1K̂_B
    K_hat = 0.3 * K_R + 0.6 * K_G + 0.1 * K_B
    
    return K_hat

def remove_fixed_pattern_noise(K_hat: np.ndarray) -> np.ndarray:
    """
    Removes fixed pattern noise (row and column) from a 2D matrix.
    
    Equation (eq3.png) implementation:
    1. r_i = 1/n * sum_{j=1}^{n} K_hat[i,j]
    2. K'[i,j] = K_hat[i,j] - r_i
    3. c_j = 1/m * sum_{i=1}^{m} K'[i,j]
    4. K''[i,j] = K'[i,j] - c_j

    """

    # Getting the dimensions of the input matrix K_hat
    m, n = K_hat.shape

    K_prime = np.copy(K_hat)
    K_double_prime = np.copy(K_prime)

    # r_i = 1/n * sum_{j=1}^{n} K_hat[i,j]
    
    r_i = np.mean(K_hat, axis=1)

    # Subtracting r_i from each row of K_hat to get K_prime.
    # K'[i,j] = K_hat[i,j] - r_i

    K_prime = K_hat - r_i[:, np.newaxis]

    #Finding the mean of each column in K_prime
    # c_j = 1/m * sum_{i=1}^{m} K'[i,j]
    
    c_j = np.mean(K_prime, axis=0)

    # Subtract c_j from each column of K_prime to get K_double_prime.
    # K''[i,j] = K'[i,j] - c_j
    
    K_double_prime = K_prime - c_j

    return K_double_prime

# --- Process all images in a folder --- #
def main(images_dir):
    image_paths = glob.glob(os.path.join(images_dir, '*.jpg'))

    R_residuals, G_residuals, B_residuals = [], [], []
    R_intensities, G_intensities, B_intensities = [], [], []
    
    for path in tqdm(image_paths, desc="Processing images"):
        img = read_image_pil(image_path)

        R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        result = compute_residual(img, R, G, B)

        
        if result is None:
            continue
        R_residual, G_residual, B_residual = result
    
        # intensity for each image (should be the original channel, not mean)
        R_Intensity = R
        G_Intensity = G
        B_Intensity = B

        R_residuals.append(R_residual)
        G_residuals.append(G_residual)
        B_residuals.append(B_residual)

        R_intensities.append(R_Intensity)
        G_intensities.append(G_Intensity)
        B_intensities.append(B_Intensity)

    # Estimate PRNU for each channel
    K_R = prnu_calculator(R_residuals, R_intensities)
    K_G = prnu_calculator(G_residuals, G_intensities)
    K_B = prnu_calculator(B_residuals, B_intensities)

    K_gray = rgb_to_grayscale(K_R, K_G, K_B)
    Fingerprint = remove_fixed_pattern_noise(K_gray)
    
    print(Fingerprint)
    

        ## Proceed to code further in this function for using the residuals for calculating PRNU (equation. 1) for each channel.
        ## In equation 1, calculate the numerator term for each channel of each image and denominator is square of each channel of each image. Resulting is the summamtion of this numerator and denominator for each channel.
        ## You get KR, KG, KB for each channel.

    ### KAVAN'S CODE

        ## Combine the PRNU of three channels to get the final PRNU using K = 0.3 * K_R + 0.6 * K_G + 0.1 * K_B
        # And then you also have to subtract the column and row artifacts from the PRNU.


if __name__ == "__main__":
    result = main("/home/kavank/D01/flat")
# main("/home/kavank/D01/flat")