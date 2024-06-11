import os
import cv2
import shutil
import numpy as np

def adjust_brightness(image, alpha):
    # 调整亮度
    new_image = np.clip(alpha * image, 0, 255).astype(np.uint8)
    return new_image

def adjust_contrast(image, alpha):
    # 调整对比度
    new_image = np.clip((1 - alpha) * image.mean() + alpha * image, 0, 255).astype(np.uint8)
    return new_image

def adjust_saturation(image, alpha):
    # 转换到HSV色彩空间并调整饱和度
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 1] = hsv[:, :, 1] * alpha
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    new_image = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2BGR)
    return new_image

# 读取图片
def read_image(image_name, save_path):
    image_path = os.path.join('datasets/endovis_wrapup/images/test_hard', image_name)
    image = cv2.imread(image_path)

    # 生成变体
    variants = []
    for alpha in np.linspace(0.5, 1.5, 2):  
        variants.append(adjust_brightness(image, alpha))
        variants.append(adjust_contrast(image, alpha))
        variants.append(adjust_saturation(image, alpha))

    new_names = []

    for i, var in enumerate(variants):
        cv2.imwrite(os.path.join(save_path, image_name.split('.')[0] + '_' + str(i) + '.png'), var)
        new_names.append(image_name.split('.')[0] + '_' + str(i) + '.png')

    return new_names

if __name__ == '__main__':
    file_list = ["17_seq_7_frame249.png",
                "17_seq_7_frame250.png",
                "17_seq_7_frame254.png",
                "17_seq_7_frame255.png",
                "17_seq_7_frame263.png",
                "17_seq_7_frame264.png",
                "17_seq_7_frame267.png",
                "17_seq_7_frame256.png",
                "17_seq_7_frame272.png",
                "17_seq_7_frame273.png",
                "17_seq_7_frame274.png",
                "17_seq_7_frame275.png",
                "17_seq_7_frame276.png",
                
                "17_seq_9_frame150.png",
                "17_seq_9_frame151.png",
                "17_seq_9_frame152.png",
                "17_seq_9_frame153.png",
                "17_seq_9_frame154.png",
                "17_seq_9_frame155.png",
                "17_seq_9_frame156.png",
                "17_seq_9_frame157.png",
                "17_seq_9_frame158.png",
                "17_seq_9_frame159.png",
                "17_seq_9_frame160.png",
                "17_seq_9_frame161.png",
                "17_seq_9_frame162.png",
                ]
    
    save_path = "aug"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_img_path = os.path.join(save_path, "images")
    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)
    save_label_path = os.path.join(save_path, "labels")
    if not os.path.exists(save_label_path):
        os.makedirs(save_label_path)

    for file in file_list:
        new_names = read_image(file, save_img_path)
        for new_name in new_names:
            shutil.copyfile(os.path.join("datasets/endovis_wrapup/labels/test_hard", file.split('.')[0] + '.txt'), os.path.join(save_label_path, new_name.split('.')[0] + '.txt'))


