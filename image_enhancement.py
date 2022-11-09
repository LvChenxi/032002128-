# -*- coding: utf-8 -*-
import cv2
import numpy as np


def PSAlgorithm(rgb_img, increment):
    img = rgb_img * 1.0
    img_min = img.min(axis=2)
    img_max = img.max(axis=2)
    img_out = img

    # 获取HSL空间的饱和度和亮度
    delta = (img_max - img_min) / 255.0
    value = (img_max + img_min) / 255.0
    L = value / 2.0

    # s = L<0.5 ? s1 : s2
    mask_1 = L < 0.5
    s1 = delta / (value)
    s2 = delta / (2 - value)
    s = s1 * mask_1 + s2 * (1 - mask_1)

    # 增量大于0，饱和度指数增强
    if increment >= 0:
        # alpha = increment+s > 1 ? alpha_1 : alpha_2
        temp = increment + s
        mask_2 = temp > 1
        alpha_1 = s
        alpha_2 = s * 0 + 1 - increment
        alpha = alpha_1 * mask_2 + alpha_2 * (1 - mask_2)

        alpha = 1 / alpha - 1
        img_out[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - L * 255.0) * alpha
        img_out[:, :, 1] = img[:, :, 1] + (img[:, :, 1] - L * 255.0) * alpha
        img_out[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - L * 255.0) * alpha

    # 增量小于0，饱和度线性衰减
    else:
        alpha = increment
        img_out[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - L * 255.0) * alpha
        img_out[:, :, 1] = img[:, :, 1] + (img[:, :, 1] - L * 255.0) * alpha
        img_out[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - L * 255.0) * alpha

    img_out = img_out / 255.0

    # RGB颜色上下限处理(小于0取0，大于1取1)
    mask_3 = img_out < 0
    mask_4 = img_out > 1
    img_out = img_out * (1 - mask_3)
    img_out = img_out * (1 - mask_4) + mask_4

    return img_out

def clahe_Img(image,ksize = 8):
    """
    :param path: 图像路径
    :param ksize: 用于直方图均衡化的网格大小，默认为8
    :return: clahe之后的图像
    """
    b, g, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(ksize,ksize))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    image = cv2.merge([b, g, r])
    return image

def hisColor_Img(img):
    """
    对图像直方图均衡化
    :param path: 图片路径
    :return: 直方图均衡化后的图像
    """
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0]) #equalizeHist(in,out)
    cv2.merge(channels, ycrcb)
    img_eq=cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
    return img_eq

def whiteBalance_Img(img):
    """
    对图像白平衡处理
    """
    b, g, r = cv2.split(img)
    Y = 0.299 * r + 0.587 * g + 0.114 * b
    Cr = 0.5 * r - 0.419 * g - 0.081 * b
    Cb = -0.169 * r - 0.331 * g + 0.5 * b
    Mr = np.mean(Cr)
    Mb = np.mean(Cb)
    Dr = np.var(Cr)
    Db = np.var(Cb)
    temp_arry = (np.abs(Cb - (Mb + Db * np.sign(Mb))) < 1.5 * Db) & (
                np.abs(Cr - (1.5 * Mr + Dr * np.sign(Mr))) < 1.5 * Dr)
    RL = Y * temp_arry
    # 选取候选白点数的最亮10%确定为最终白点，并选择其前10%中的最小亮度值
    L_list = list(np.reshape(RL, (RL.shape[0] * RL.shape[1],)).astype(np.int))
    hist_list = np.zeros(256)
    min_val = 0
    sum = 0
    for val in L_list:
        hist_list[val] += 1
    for l_val in range(255, 0, -1):
        sum += hist_list[l_val]
        if sum >= len(L_list) * 0.1:
            min_val = l_val
            break
    # 取最亮的前10%为最终的白点
    white_index = RL < min_val
    RL[white_index] = 0
    # 计算选取为白点的每个通道的增益
    b[white_index] = 0
    g[white_index] = 0
    r[white_index] = 0
    Y_max = np.max(RL)
    b_gain = Y_max / (np.sum(b) / np.sum(b > 0))
    g_gain = Y_max / (np.sum(g) / np.sum(g > 0))
    r_gain = Y_max / (np.sum(r) / np.sum(r > 0))
    b, g, r = cv2.split(img)
    b = b * b_gain
    g = g * g_gain
    r = r * r_gain
    # 溢出处理
    b[b > 255] = 255
    g[g > 255] = 255
    r[r > 255] = 255
    res_img = cv2.merge((b, g, r))
    return res_img

if __name__=='__main__':
    enhance_img = cv2.imread(r"./img/3.jpg")
    enhance_img = clahe_Img(enhance_img)
    enhance_img = whiteBalance_Img(enhance_img)
    cv2.imwrite(r"./img/enhanced_2.jpg", enhance_img)
