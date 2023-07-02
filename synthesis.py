import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
import matplotlib.patches as patches
import cv2
import numpy as np
from tqdm import tqdm
import csv


def AugmentImage(img, mode=0):
    assert mode >= 0 and mode <= 7

    mode0 = mode % 4 # rotate(0-no, 1-90 deg, 2-180 deg，3-270 deg)
    mode1 = mode // 4 # # flip(0-no，1-updown flip)

    imgr = np.rot90(img, mode0)
    if mode1 == 0:
        imgf = imgr
    else:
        imgf = np.flip(imgr, axis=0)

    return imgf


def transmission(w, cirrus_file_name, cirrus_aug=0, GT_dir="", mode=0):
    assert w >= 0 and w <= 1

    if mode == 1:
        try:
            img = cv2.imread(cirrus_file_name, cv2.IMREAD_ANYDEPTH)
            # print('r1: {}'.format(img.shape))
            img = (img - img.min()) / (img.max() - img.min())           # min-max normalize
        except:
            print("[ERROR]: Fail to open tif file: {}.".format(cirrus_file_name))
            exit(0)
    else:
        try:
            img = cv2.imread(cirrus_file_name, cv2.IMREAD_ANYDEPTH)
            # print('r1: {}'.format(img.shape))
            img = img / img.max()           # max normalize
        except:
            print("[ERROR]: Fail to open tif file: {}.".format(cirrus_file_name))
            exit(0)

    img_aug = AugmentImage(img, cirrus_aug)
    t1_img = 1.0 - w * img_aug
    return t1_img


def gamma(t_img, mode):
    if mode==0:
        a = 0.4318
        b = -1.896
        tao_l = 0.285
        tao_h = 0.851

        # t_img = np.clip(t_img, tao_l, tao_h)
        gamma_img = np.clip(a + b * np.log(np.log(1/t_img)), 0, 4)
    else:
        t_img = 1 - t_img 
        a3 = -21.547
        a2 = 41.224
        a1 = -27.465
        a0 = 6.537
        gamma_img = np.clip(a3*t_img**3 + a2*t_img**2 + a1*t_img + a0, 0, 4)
        # gamma_img = np.ones((t_img.shape[0], t_img.shape[1]))


    return gamma_img


def AtmosphericLight(H_file):
    try:
        H_j = cv2.imread(H_file, cv2.IMREAD_ANYDEPTH)
        # print('r2: {}'.format(H_j.shape))
        H_j[H_j > 2**15-1] = 0
        H_j = cv2.resize(H_j, (1000, 1000)) # downsample for speedup
        scale = 10000                       # data stored in [0, 10000]
        H_j = H_j / scale
        H_j = np.clip(H_j, 0, 1)    # normalize
    except:
        print("[ERROR]: Fail to open tif file: {}.".format(H_file))
        exit(0)

    h, w = H_j.shape[:2]
    l = h * w

    # slice top 0.01% pixels
    _H_j = H_j.flatten()
    numpx = max(l // 10000, 1)
    indices = np.argsort(_H_j)[-numpx:]
    sliced_H_j = _H_j[indices]

    A_j = sliced_H_j.mean(0)                # mean of top 0.01% pixels

    return A_j


def AddHazeGT(J_path, A, t1_img, gamma_img, GT_dir="", Hazy_dir=""):
    waveLens = [0.4430, 0.4825, 0.5625, 0.6550, 0.8650, 1.6100, 2.2000]
    waveLen1 = waveLens[0]
    waveLen2 = waveLens[1] 

    J_list = os.listdir(J_path)
    J_head  = J_path.split('/')[-1]
    if len(J_list)<9:
        print("[ERROR]: Unexpected number of files: {} in {}.".format(len(J_list), J_path))

    I = []
    J = []

    for b in range(1, 8):
        waveLen_j = waveLens[b-1]
        waveLen1_j = waveLen1 / waveLen_j
        J_file = J_path + '/' + J_head + '_B' + str(b) + '.png'

        A_j = A[b-1]
        # print("[INFO]: A of band {} is {}.".format(b,A_j))

        try:
            J_j = cv2.imread(J_file, cv2.IMREAD_ANYDEPTH)
            # print('r3: {}'.format(J_j.shape))
            J_j[J_j > 2**16-1] = 0
            scale = 65535               # data stored in [0, 65535]
            J_j = J_j / scale
            J_j = np.clip(J_j, 0, 1)    # normalize
        except:
            print("[ERROR]: Fail to open tif file: {}.".format(J_file))
            exit(0)

        J_j_path = GT_dir + '/' + GT_dir.split('/')[-1] + '_B{}.png'.format(b)
        # if os.path.exists(J_j_path):
        #     print("[WARNING]: Existing image: {}.".format(J_j_path))
        # cv2.imwrite(J_j_path, (J_j * 65535).astype(np.uint16))                      # 65535 for 16-bit image

        # 加雾
        if gamma_img.shape != J_j.shape:
            print("[ERROR]: Incompatible dimension between 'gamma'{} and 'J'{}."\
                .format(gamma_img.shape, J_j.shape))

        t_j = np.exp(np.multiply(np.power(waveLen1_j, gamma_img), np.log(t1_img)))  # t1 to other t
        I_j = J_j * t_j + A_j * (1 - t_j)                                           # formulation

        t_j_path = Hazy_dir + '/' + 't_{}.png'.format(b)
        cv2.imwrite(t_j_path, ((1-t_j) * 65535).astype(np.uint16)) 

        I_j_path = Hazy_dir + '/' + Hazy_dir.split('/')[-1] + '_B{}.png'.format(b)
        # if os.path.exists(I_j_path):
        #     print("[WARNING]: Existing image: {}.".format(J_j_path))
        cv2.imwrite(I_j_path, (I_j * 65535).astype(np.uint16))                      # 65535 for 16-bit image
        # print('w1: {}'.format(I_j.shape))

        # BGR for visualization
        if b in [2, 3, 4]:
            I.append(I_j)
            J.append(J_j)

    I = np.stack(I, axis=2)
    J = np.stack(J, axis=2)

    I_RGB_path = Hazy_dir + '/' + Hazy_dir.split('/')[-1] + '_RGB.png'
    cv2.imwrite(I_RGB_path, (I * 255).astype(np.uint8))

    J_RGB_path = GT_dir + '/' + GT_dir.split('/')[-1] + '_RGB.png'
    # cv2.imwrite(J_RGB_path, (J * 255).astype(np.uint8))

    I = np.power(I, 1/2.2)
    J = np.power(J, 1/2.2)

    I_RGB_path = Hazy_dir + '/' + Hazy_dir.split('/')[-1] + '_RGB_gamma.png'
    cv2.imwrite(I_RGB_path, (I * 255).astype(np.uint8))

    J_RGB_path = GT_dir + '/' + GT_dir.split('/')[-1] + '_RGB_gamma.png'
    # cv2.imwrite(J_RGB_path, (J * 255).astype(np.uint8))


def Generate_w(segment=0, mode=0):
    assert(mode>=0)

    if mode == 0: # random in [0.001, 0.1), [0.1, 0.2) ...  [0.9, 1.0)
        assert(segment>=0 and segment<10)
        w = np.random.randint(segment*100+1, (segment+1)*100+1)
        w = w / 1000.0

    elif mode == 1: # log in [0.05, 1.0)
        min_log = np.log(0.05)
        max_log = np.log(1)

        log_w = min_log + np.random.rand() * (max_log - min_log)
        w = np.exp(log_w)

    elif mode == 2: # gassian in [0, 0.333], [0.334, 0.666], [0.667, 1.000)
        assert(segment>=0 and segment<3)
        if segment == 0:
            w = np.random.normal(loc=0.1665, scale=0.0500)
            if w < 0:
                w = 0
            if w > 0.3333:
                w = 0.3333
        elif segment == 1:
            w = np.random.normal(loc=0.5000, scale=0.0500)
            if w < 0.3334:
                w = 0.3334
            if w > 0.6666:
                w = 0.6666
        else:
            w = np.random.normal(loc=0.8335, scale=0.0500)
            if w < 0.6667:
                w = 0.6667
            if w > 1.0000:
                w = 1.0000

    else: # random in [0, 1.0)
        w = np.random.randint(1, 1000)
        w = w / 1000.0
    
    return w


def SaveRGB(J_path, GT_dir):
    J = []
    J_head  = J_path.split('/')[-1]
    for b in range(2,5):
        J_file = J_path + '/' + J_head + '_B' + str(b) + '.tiff'
        print('J_{} tiff: {}.'.format(b, J_file))
        try:
            J_j = cv2.imread(J_file, cv2.IMREAD_ANYDEPTH)
            # print('r3: {}'.format(J_j.shape))
            J_j[J_j > 2**16-1] = 0      # change negative to zero
            scale = 65536               # data stored in [0, 65535]
            J_j = J_j / scale
            J_j = np.clip(J_j, 0, 1)    # normalize
        except:
            print("[ERROR]: Fail to open tif file: {}.".format(J_file))
            exit(0)

        J.append(J_j)
    J = np.stack(J, axis=2)

    J_RGB_path = GT_dir + '/' + GT_dir.split('/')[-1] + '_RGB.png'
    cv2.imwrite(J_RGB_path, (J * 255).astype(np.uint8))

    J = np.power(J, 1/2.2)

    J_RGB_path = GT_dir + '/' + GT_dir.split('/')[-1] + '_RGB_gamma.png'
    cv2.imwrite(J_RGB_path, (J * 255).astype(np.uint8))


def CreateHazyImage(clear_img, cloud_dir, hazy_dir):
    if not os.path.exists(clear_img):
        print("[ERROR]: Invalid path for clear multiband RS image: {}."\
            .format(clear_img))
    if not os.path.exists(cloud_dir):
        print("[ERROR]: Invalid path for cloud image: {}.".format(cloud_dir))

    cirrus_dir = os.listdir(cloud_dir)
    cirrus_dir.sort()
    cirrus_num = len(cirrus_dir)
    
    A_av = [0.4675584415584417, 0.45558441558441565, 0.4893246753246754, 0.5146753246753245, 0.6082207792207793, 0.575025974025974, 0.5434935064935067] # 77张
    A_ref = (A_av[5] + A_av[6]) / 2
    A_ratio = [a/A_ref for a in A_av]

    A_path = clear_img + '.csv'
    A_csv = csv.reader(open(A_path, 'r'))
    for line in A_csv:
        A = [float(line[-7]), float(line[-6]), float(line[-5]), float(line[-4]), float(line[-3]), float(line[-2]), float(line[-1])]
    A_ref = (A[5] + A[6]) / 2
    A_fix = [(A_ref*r) if A_ref*r<1.0 else 1.0 for r in A_ratio]
    alpha = 0.1
    A_fix = [alpha*a+(1-alpha)*b if alpha*a+(1-alpha)*b<1.0 else 1.0  for a,b in zip(A, A_fix)]
    
    print("A before fixing: {}".format(A))
    print("A after fixing: {}".format(A_fix))
    # A before fixing: [0.256, 0.257, 0.297, 0.338, 0.536, 0.537, 0.588]
    # A after fixing:  [0.4488406590344379, 0.43810159765924345, 0.47264376843229683, 0.4996915426235978, 0.6041702401133224, 0.5742218226784014, 0.5507781773215986]

    # add haze and save image
    for i in range(1): 
        # w = Generate_w(mode=1) # log sample
        w = 0.502
        curHazy_dir = hazy_dir + '/' + hazy_dir.split('/')[-1] + \
                        "_{}".format(i) + "_{:.4f}".format(w) + "_minmax"
        os.makedirs(curHazy_dir, exist_ok=True)
        curHazy_dir_old = hazy_dir + '/' + hazy_dir.split('/')[-1] + \
                        "_{}".format(i) + "_{:.4f}".format(w) + "_max"
        os.makedirs(curHazy_dir_old, exist_ok=True)

        # pick cirrus image randomly
        cirrus_seed = np.random.randint(0, cirrus_num)
        cirrus_aug = np.random.randint(0, 8) # rotate and flip
        tail = cirrus_dir[cirrus_seed] + '/' + \
                cirrus_dir[cirrus_seed] + '_B9.tiff'
        cirrusband_path = cloud_dir + '/' + tail

        # add haze
        t1_img0 = transmission(w, cirrusband_path, cirrus_aug, mode=0)
        t1_img1 = transmission(w, cirrusband_path, cirrus_aug, mode=1)
        # SaveRGB(clear_img, curHazy_dir_old)
        gamma_img0 = gamma(t1_img0, mode=1)
        gamma_img1 = gamma(t1_img1, mode=1)

        AddHazeGT(clear_img, A_fix, t1_img0, gamma_img0, "", curHazy_dir_old)
        AddHazeGT(clear_img, A_fix, t1_img1, gamma_img1, "", curHazy_dir)



if __name__ == "__main__":
    clear_img = r"./data/00215/00215"
    cloud_dir = r"./data/cirrus/"
    hazy_dir = r"./result/00215"
    CreateHazyImage(clear_img, cloud_dir, hazy_dir)