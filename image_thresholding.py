
import numpy as np
from PIL import Image
from IPython.display import Image as IPImage, display
import os
import matplotlib.pyplot as plt



from scipy.signal import find_peaks 
from scipy.signal import peak_prominences 


import argparse


def preprocess_single_image(image_path): 
    image = Image.open(image_path)
    gray_image = image.convert('L')
    image_array = np.array(gray_image)
    return image_array

def preprocess_images_from_directory(directory_path):
    pre_processed_images = []
    filenames = []
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            file_path = os.path.join(directory_path, filename)
            image_array = preprocess_single_image(file_path)
            pre_processed_images.append(image_array)
            filenames.append(filename)
    return pre_processed_images,filenames



def mean_thresholding(image_array,window_size,step_size):

  #This function applies mean thresholding to an image by sliding a window
  #of size window_size across the image with a step size of step_size. 
  #The threshold for each window is calculated as the mean pixel intensity within the window. 
  #If the intensity of the center pixel of the window exceeds the threshold, 
  #the corresponding pixel in the output image is set to 0 (black) otherwise, it is set to 255 (white).
  
  original_shape = image_array.shape
  y_max = original_shape[0]
  x_max = original_shape[1]
  output_array = np.zeros(original_shape, dtype=np.uint8)

  pad_size = int(window_size/2)
  padded_img_array = np.pad(image_array, pad_width=pad_size, mode='edge')

  for y in range(pad_size, y_max+pad_size, step_size):
    for x in range(pad_size,x_max+pad_size, step_size):
        window = padded_img_array[y-pad_size:y+pad_size, x-pad_size:x+pad_size]
        threshold = np.mean(window)
        if(padded_img_array[y,x] > threshold):
          output_array[y-pad_size,x-pad_size] = 0
        else:
          output_array[y-pad_size,x-pad_size] = 255
  return output_array


def calculate_interclass_variance(img_array,threshold,histogram): 

    #calculates interclass variance for a given histogram and threshold for otsus_method function

    height, width = img_array.shape[:2]
    total_pixels = height * width

    mask_bg = img_array < threshold
    mask_fg = img_array > threshold

    background_count = np.sum(mask_bg)
    foreground_count = np.sum(mask_fg)

    W_b = background_count/total_pixels
    W_f = foreground_count/total_pixels

    intensity_levels = np.arange(256)
    numerator_b = np.sum(intensity_levels[:threshold]* histogram[:threshold])
    mu_b = numerator_b/background_count

    numerator_f =  np.sum(intensity_levels[threshold:]* histogram[threshold:])
    mu_f = numerator_f/foreground_count
    
    class_variance = W_b*W_f*((mu_b-mu_f)**2)
    return class_variance


def otsus_method(img_array):
  
  #creates the Grayscale image histogram
  #finds peaks with the restrictions on their horizontal distances inbetween and minimum heights
  #chooses the most prominent peaks
  #computes the interclass variance for five values between the corresponding intensity values of the most prominent peaks
  #takes as threshold the intensity value that gives the highest interclass variance

  histogram, bin_edges = np.histogram(img_array, bins=256, range=(0, 255))

  max_count = np.max(histogram)
  min_height = max_count/6

  plt.plot(bin_edges[0:-1], histogram)
  plt.title('Grayscale Image Histogram')
  plt.xlabel('Pixel Intensity')
  plt.ylabel('Pixel Count')
  plt.show()
  horizontal_distance = 25

  peaks, _ = find_peaks(histogram,distance = horizontal_distance, height=min_height)

  if len(peaks) == 1 :
    peaks, _ = find_peaks(histogram)
  
  if len(peaks) == 0 : 
    threshold = 125
    
  else:
    prominences = peak_prominences(histogram,peaks)[0]
    
    sorted_indices = np.argsort(prominences)[::-1]
    top_two_peaks = peaks[sorted_indices[:2]]

    peak1 = top_two_peaks[0]
    peak2 = top_two_peaks[1]
    
    possible_thresholds = np.linspace(peak1, peak2, 7)[1:-1]
    variances = {}
    for val in possible_thresholds:
      variances[val] = calculate_interclass_variance(img_array,int(val),histogram)
    threshold = max(variances,key = variances.get )
  binary_image = np.where(img_array>threshold , 255,0)
  binary_image = binary_image.astype('uint8')

  
  return binary_image


def main(input_path,directory_path,threshold_type):

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")
    single_image = False
    if input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        pre_images = [preprocess_single_image(input_path)]
        single_image = True
        filenames = [input_path]
    elif os.path.isdir(input_path): 
        pre_images,filenames = preprocess_images_from_directory(input_path)
    thresholded_images = []
    for pre_image,filename in zip(pre_images,filenames):
        if threshold_type == 'adaptive':
            output_array = mean_thresholding(pre_image,16,1)
        elif threshold_type =='global':
            output_array = otsus_method(pre_image)
        thresholded_images.append(output_array)
        final_image = Image.fromarray(output_array)
        orig_name_wo_ext, ext = os.path.splitext(filename)
        new_filename = orig_name_wo_ext + '_processed' + ext
        file_path = os.path.join(directory_path,new_filename)
        final_image.save(file_path)
    if single_image:
        display(IPImage(file_path))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='Path to the input file or directory containing images')
    parser.add_argument('directory_path', help='Path to the directory for output images')
    parser.add_argument('threshold_type', help='Type of threshold: "global" or "adaptive" ')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main(args.input_path, args.directory_path, args.threshold_type)



