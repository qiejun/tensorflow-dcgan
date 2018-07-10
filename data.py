import numpy as np
import os
import cv2

#将图像缩放到（-1,1）
def scale(x):
    x = x/255*2-1
    return x


def load_data(imgs_path, batch_size):
    """
    :param imgs_path: 训练图片路径
    :param batch_size:
    :return: batch imgs
    """
    datas_list = []
    imgs_list = os.listdir(imgs_path)
    for img_name in imgs_list:
        img_path = os.path.join(imgs_path, img_name)
        datas_list.append(img_path)
    np.random.shuffle(datas_list)
    batch_num = len(datas_list) // batch_size
    for i in range(batch_num):
        batch_datas = datas_list[i * batch_size:(i + 1) * batch_size]
        batch_imgs = []
        for img_path in batch_datas:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (64, 64))
            img = scale(img)
            batch_imgs.append(img)
        yield np.array(batch_imgs)
