import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from scipy import stats

np.set_printoptions(threshold=np.nan)

# The path of training images
path = '/Users/ljq/Desktop/whale_data/train/'
file_list = os.listdir(path)
# The path of saving converted images
os.chdir('/Users/ljq/Desktop/whale_data/converted/')
# path = '/data8t/ljq/whale_data/whale_data/train/'
# file_list = os.listdir(path)
# os.chdir('/data8t/ljq/whale_data/whale_data/convert/converted')

# Function to find the top 5 modes in a np.ndarray
def find_modes(mat, num):
    m, n = np.shape(mat)
    count_dict = {}
    for i in range(m):
        for j in range(n):
            if (mat[i][j] in count_dict.keys()):
                count_dict[mat[i][j]] += 1
            else:
                count_dict[mat[i][j]] = 0
    max_list = []
    for k in range(num):
        temp = 0
        temp_key = '0'
        for key in count_dict.keys():
            if (count_dict[key] > temp):
                temp = count_dict[key]
                temp_key = key
            else:
                pass
        max_list.append(temp_key)
        if (len(count_dict.keys()) > 1):
            del count_dict[temp_key]
        else:
            break
            #         print(count_dict)

    return max_list

# Function to find the marigin of maximum connected area
def find_margins(mat,num):
    m,n = np.shape(mat)
    up,left = 100000,100000
    down,right = 0,0
    for i in range(m):
        for j in range(n):
            if(mat[i][j] == num):
                if(i < up):
                    up = i
                if(i > down):
                    down = i
                if(j < left):
                    left = j
                if(j > right):
                    right = j
            else:
                pass
    # Try to enlarge the cutting are
    # Remenber to try from large value to smaller one
    if(up > 50):
        up = up - 50
    elif(up > 20):
        up = up - 20
    elif(up > 10):
        up = up - 10
    if(down < m -1 -50):
        down = down + 50
    elif(down < m -1 -20):
        down = down + 20
    elif(down < m -1 -10):
        down = down + 10
    if(left > 50):
        left = left + 50
    elif(left > 20):
        left = left + 20
    elif(left > 10):
        left = left + 10
    if(right < n -1 -50):
        right = right + 50
    elif(right < n -1 -20):
        right = right + 20
    elif(right < n -1 -10):
        right = right + 10
    return up,down,left,right

# In MAC, the 1st file is ".DS_Store"
for i in range(1, len(file_list)):
    print("file name: ", file_list[i])
    # Loading images
    img = cv2.imread(path + file_list[i])
    # Image Denoising
    fil_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    gray_img = cv2.cvtColor(fil_img, cv2.COLOR_BGR2GRAY)
    # Size of image
    height = gray_img.shape[0]
    width = gray_img.shape[1]
    mask = np.zeros(gray_img.shape, np.uint8)
    # Histogram equalization
    equ_img = cv2.equalizeHist(gray_img)
    # Obtain the threshold
    hist = np.histogram(equ_img.ravel(), bins=2000)
    sum_temp = 0
    for j in range(len(hist[0])):
        sum_temp += hist[0][j]
        if (sum_temp > sum(hist[0]) / 5 * 2):
            thresh = hist[1][j]
            break
    print("threshold: ", thresh)
    # Threshold segmentation
    ret, bin_img = cv2.threshold(equ_img, thresh, 255, cv2.THRESH_BINARY)
    # Reverse (black<->white)
    bin_img = 255 - bin_img
    # Erosion to denoise
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(bin_img, kernel, iterations=2)
    # Connected-domain Analysis
    _, labels = cv2.connectedComponents(erosion)
    # print(labels)

    # Select central area
    cut_factor = 4
    #     print(round(height/cut_factor))
    #     print(round(height-height/cut_factor))
    #     print(round(width/cut_factor))
    #     print(round(width-width/cut_factor))
    search_label = labels[round(height / cut_factor):round(height - height / cut_factor),
                   round(width / cut_factor):round(width - width / cut_factor)]
    # Find the index of the largest area "connect_mode"
    connect_nums = stats.mode(search_label, axis=None)[0][0]
    if (connect_nums == 0):
        connect_index = find_modes(search_label, 2)
        connect_mode = connect_index[-1]
    else:
        connect_mode = connect_nums
    print("connected area index: ", connect_mode)

    # Using mask to test the performance
    mask = np.where(labels == connect_mode, 1, 0).astype('uint8')
    # print(mask)
    new_img = bin_img * mask
    #     # Visualization of binary images
    #     plt.subplot(131)
    #     plt.imshow(bin_img)
    #     plt.subplot(132)
    #     plt.imshow(new_img)
    #     plt.subplot(133)
    #     plt.imshow(search_label)
    #     plt.show()

    # Cut the original images
    up, down, left, right = find_margins(labels, connect_mode)
    img_cut = img[up:down, left:right]
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(img_cut)
    plt.savefig(file_list[i])
    plt.show()