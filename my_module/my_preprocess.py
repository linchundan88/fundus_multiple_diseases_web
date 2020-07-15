'''
眼底图像预处理模块
del_black_or_white 删除视网膜图像的边缘(黑色，或者白色)

detect_xyr  hough检测圆，如果检测不到，假设中心，半径
my_crop_xyz  根据检测的圆，进行裁剪
add_black  四周增加一些黑边，img aug时候 crop,rotate不会删除有意义区域

load_resize_images 加载图像 返回list，给my_img_aug使用

'''

import cv2
import numpy as np
import os
import imgaug as ia
from imgaug import augmenters as iaa


# 最短边长 乘以 ratio  已经resize 1200 左右了
DEL_PADDING_RATIO = 0.02  # used for del_black_or_white
CROP_PADDING_RATIO = 0.02  # used for my_crop_xyr

# 像素阈值
THRETHOLD_LOW = 7
THRETHOLD_HIGH = 180

# 检测圆最小半径，最大半径 r/2-r/minRedius
MIN_REDIUS_RATIO = 0.33
MAX_REDIUS_RATIO = 0.6

#illegal image
IMG_SMALL_THRETHOLD = 80

def del_black_or_white(img1):
    if img1.ndim == 2:
        img1 = np.expand_dims(img1, axis=-1)

    width, height = (img1.shape[1], img1.shape[0])

    (left, bottom) = (0, 0)
    (right, top) = (width, height)

    padding = int(min(width, height) * DEL_PADDING_RATIO)


    for i in range(width):
        array1 = img1[:, i, :]  # array1.shape[1]=3 RGB
        if np.sum(array1) > THRETHOLD_LOW * array1.shape[0] * array1.shape[1] and \
                np.sum(array1) < THRETHOLD_HIGH * array1.shape[0] * array1.shape[1]:
            left = i
            break
    left = max(0, left - padding)  # 留一些空白

    for i in range(width - 1, 0 - 1, -1):
        array1 = img1[:, i, :]
        if np.sum(array1) > THRETHOLD_LOW * array1.shape[0] * array1.shape[1] and \
                np.sum(array1) < THRETHOLD_HIGH * array1.shape[0] * array1.shape[1]:
            right = i
            break
    right = min(width, right + padding)  # 留一些空白

    for i in range(height):
        array1 = img1[i, :, :]
        if np.sum(array1) > THRETHOLD_LOW * array1.shape[0] * array1.shape[1] and \
                np.sum(array1) < THRETHOLD_HIGH * array1.shape[0] * array1.shape[1]:
            bottom = i
            break
    bottom = max(0, bottom - padding)

    for i in range(height - 1, 0 - 1, -1):
        array1 = img1[i, :, :]
        if np.sum(array1) > THRETHOLD_LOW * array1.shape[0] * array1.shape[1] and \
                np.sum(array1) < THRETHOLD_HIGH * array1.shape[0] * array1.shape[1]:
            top = i
            break
    top = min(height, top + padding)

    img2 = img1[bottom:top, left:right, :]

    return img2

def detect_xyr(img_source):
    if isinstance(img_source, str):
        try:
            img1 = cv2.imread(img_source)
        except:
            # Corrupt JPEG data1: 19 extraneous bytes before marker 0xc4
            raise Exception("image file not found:" + img_source)
    else:
        img1 = img_source

    if img1 is None:
        raise Exception("image file error:" + img_source)

    width, height = (img1.shape[1], img1.shape[0])

    myMinWidthHeight = min(width, height)  # 最短边长1600 宽和高的最小,并不是所有的图片宽>高 train/22054_left.jpeg 相反
    myMinRadius = round(myMinWidthHeight * MIN_REDIUS_RATIO)
    myMaxRadius = round(myMinWidthHeight * MAX_REDIUS_RATIO)

    '''
    minDist：检测到的圆的中心，（x,y）坐标之间的最小距离。如果minDist太小，则可能导致检测到多个相邻的圆。如果minDist太大，则可能导致很多圆检测不到。
    minDist表示两个圆之间圆心的最小距离
    param1：用于处理边缘检测的梯度值方法。
    param2：cv2.HOUGH_GRADIENT方法的累加器阈值。阈值越小，检测到的圈子越多。

    According to our test about fundus images, param2 = 30 is enough, too high will miss some circles
    '''
    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=450, param1=120, param2=60,
    #                            minRadius=myMinRadius,
    #                            maxRadius=myMaxRadius)

    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=450, param1=120, param2=32,
                               minRadius=myMinRadius, maxRadius=myMaxRadius)

    found_circle = False

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        if (circles is not None) and (len(circles == 1)):
            # 有些圆心位置很离谱 25.Hard exudates/chen_liang quan_05041954_clq19540405_557410.jpg

            x, y, r = circles[0]
            if x > (2 / 5 * width) and x < (3 / 5 * width) \
                    and y > (2 / 5 * height) and y < (3 / 5 * height):
                found_circle = True

    if not found_circle:
        # suppose the center of the image is the center of the circle.
        x = img1.shape[1] // 2
        y = img1.shape[0] // 2

        # get radius  according to the distribution of pixels of the middle line
        temp_x = img1[int(img1.shape[0] / 2), :, :].sum(1)
        r = int((temp_x > temp_x.mean() / 12).sum() / 2)

    return (found_circle, x, y, r)

def my_crop_xyr(img_source, x, y, r, crop_size=None):
    if isinstance(img_source, str):
        try:
            img1 = cv2.imread(img_source)
        except:
            # Corrupt JPEG data1: 19 extraneous bytes before marker 0xc4
            raise Exception("image file not found:" + img_source)
    else:
        img1 = img_source

    if img1 is None:
        raise Exception("image file error:" + img_source)

    height, width = img1.shape[:2]

    #  裁剪图像 根据半径裁减  判断高是否够  防止超过边界,所以0和width
    # 第一个是高,第二个是宽  r是半径

    img_padding = int(min(width, height) * CROP_PADDING_RATIO)

    image_left = int(max(0, x - r - img_padding))
    image_right = int(min(x + r + img_padding, width - 1))
    image_bottom = int(max(0, y - r - img_padding))
    image_top = int(min(y + r + img_padding, height - 1))

    if width >= height:  # 图像宽比高大
        if height >= 2 * (r + img_padding):
            # 图像比圆大
            img1 = img1[image_bottom: image_top, image_left:image_right]
        else:
            # 因为图像高度不够,图像被垂直剪切
            img1 = img1[:, image_left:image_right]
    else:  # 图像宽比高小
        if width >= 2 * (r + img_padding):
            # 图像比圆大
            img1 = img1[image_bottom: image_top, image_left:image_right]
        else:
            img1 = img1[image_bottom:image_top, :]

    if crop_size is not None:
        img1 = cv2.resize(img1, (crop_size, crop_size))

    return img1

# 给图像增加一些黑边， imgaug旋转，crop 防止 剪切掉有意义的区域
def add_black_margin(img_source, add_black_pixel_ratio=0.05):
    if isinstance(img_source, str):
        try:
            image1 = cv2.imread(img_source)
        except:
            # Corrupt JPEG data1: 19 extraneous bytes before marker 0xc4
            raise Exception("image file not found:" + img_source)
    else:
        image1 = img_source

    if image1 is None:
        raise Exception("image file error:" + img_source)

    height, width = image1.shape[:2]

    add_black_pixel = int(min(height, width) * add_black_pixel_ratio)

    img_h = np.zeros((add_black_pixel, width, 3))
    img_v = np.zeros((height + add_black_pixel * 2, add_black_pixel, 3))

    image1 = np.concatenate((img_h, image1, img_h), axis=0)
    image1 = np.concatenate((img_v, image1, img_v), axis=1)

    return image1


# 训练的时候 add_black_pixel=40, valid(test)时候 add_black_pixel=0
# 给图像增加一些黑边， imgaug旋转，crop 防止 剪切掉有意义的区域
def my_preprocess(img_source, crop_size, add_black_pixel_ratio=0, img_file_dest=None):
    if isinstance(img_source, str):
        try:
            img1 = cv2.imread(img_source)
        except:
            # Corrupt JPEG data1: 19 extraneous bytes before marker 0xc4
            raise Exception("image file not found:" + img_source)
    else:
        img1 = img_source

    if img1 is None:
        raise Exception("image file error:" + img_source)

    # 删除黑色部分
    img1 = del_black_or_white(img1)

    # 删除黑色以后，图像大小可能差别比较大,   太大的图片，缩放一下，再检测圆
    min_width_height = min(img1.shape[:2])

    if min_width_height < IMG_SMALL_THRETHOLD:  # image too small
        return None

    resize_size = crop_size * 2.5
    if min_width_height > resize_size:
        crop_ratio = resize_size / min_width_height
        img1 = cv2.resize(img1, None, fx=crop_ratio, fy=crop_ratio)

    (found_circle, x, y, r) = detect_xyr(img1)

    if add_black_pixel_ratio > 0:
        img1 = my_crop_xyr(img1, x, y, r)
        # add some black margin, for fear that duing img aug(random rotate crop) delete useful areas
        img1 = add_black_margin(img1, add_black_pixel_ratio=add_black_pixel_ratio)
        img1 = cv2.resize(img1, (crop_size, crop_size))
    else:
        img1 = my_crop_xyr(img1, x, y, r, crop_size)

    if img_file_dest is not None:
        os.makedirs(os.path.dirname(img_file_dest), exist_ok=True)
        cv2.imwrite(img_file_dest, img1)

    return img1

# 预测的时候 multi crop
def multi_crop(img_source, gen_times=5, add_black=True):
    if isinstance(img_source, str):
        try:
            image1 = cv2.imread(img_source)
        except:
            # Corrupt JPEG data1: 19 extraneous bytes before marker 0xc4
            raise Exception("image file not found:" + img_source)
    else:
        image1 = img_source

    if image1 is None:
        raise Exception("image file error:" + img_source)

    if add_black:
        image1 = add_black(img_source)

    list_image = [image1]

    # sometimes = lambda aug: iaa.Sometimes(0.96, aug)
    seq = iaa.Sequential([
        iaa.Crop(px=(0, min(image1.shape[0], image1.shape[1]) // 20)),
        # crop images from each side by 0 to 16px (randomly chosen)
        # iaa.GaussianBlur(sigma=(0, 3.0)),  # blur images with a sigma of 0 to 3.0,
        # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
        # sometimes(iaa.Crop(percent=(0, 0.1))),  # crop images by 0-10% of their height/width
        # shuortcut for CropAndPad

        # improve or worsen the contrast  If PCH is set to true, the process happens channel-wise with possibly different S.
        # sometimes1(iaa.ContrastNormalization((0.9, 1.1), per_channel=0.5), ),
        # change brightness of images (by -5 to 5 of original value)
        # sometimes1(iaa.Add((-6, 6), per_channel=0.5),),
        # sometimes(iaa.Affine(
        #     # scale={"x": (0.92, 1.08), "y": (0.92, 1.08)},
        #     # scale images to 80-120% of their size, individually per axis
        #     # Translation Shifts the pixels of the image by the specified amounts in the x and y directions
        #     translate_percent={"x": (-0.08, 0.08), "y": (-0.06, 0.06)},
        #     # translate by -20 to +20 percent (per axis)
        #     rotate=(0, 360),  # rotate by -45 to +45 degrees
        #     # shear=(-16, 16),  # shear by -16 to +16 degrees
        #     # order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
        #     # cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
        #     # mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        # )),
    ])

    img_results = []

    for i in range(gen_times):
        images_aug = seq.augment_images(list_image)
        img_results.append(images_aug[0])

    return img_results

#test code
if __name__ == '__main__':
    img_file = '/home/jsiec/disk1/PACS/公开数据集/IDRID/test/original/IDRiD_32.jpg'

    d = multi_crop(img_file, 5)

    # img_file = '/tmp/image1.jpg'
    img1 = cv2.imread(img_file)

    img2 = del_black_or_white(img1)

    # img2 = my_preprocess(img1, 512)

    cv2.imwrite('/tmp/01.jpg', img2)

    exit(0)

    img3 = add_black_margin(img2)

    cv2.imwrite('/tmp/02.jpg', img3)

    pass