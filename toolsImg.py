import os
from PIL import Image
import random
import cv2 


def load_img(img_path):
    try:
        img = Image.open(img_path)
        return img
    except Exception as e:
        print("Error while loading image: ", img_path, " ", e)
        return None

def is_img_file(file_path):
    extensions = (".jpg", ".jpeg", ".png", ".gif")
    return file_path.lower().endswith(extensions)

def get_img_list(folder_path):
    img_list = []
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        filenames = os.listdir(folder_path)
        for filename in filenames:
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and is_img_file(file_path):
                img = load_img(file_path)
                img_list.append(img)
    return img_list 
 
def resize_image(image, size):
    return cv2.resize(image, (size, size))


def normalize_image(image):
    return image / 255.0

def augment_image(image):
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
    
    
    if random.random() > 0.5:
        image = cv2.flip(image, 0)
    
    
    if random.random() > 0.5:
        angle = random.randint(-30, 30)
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))

    
    if random.random() > 0.5:
        h, w = image.shape[:2]
        top = random.randint(0, h // 4)
        left = random.randint(0, w // 4)
        bottom = random.randint(3 * h // 4, h) 
        right = random.randint(3 * w // 4, w)
        image = image[top:bottom, left:right]
        image = cv2.resize(image, (w, h))

    if random.random() > 0.5:
        value = random.uniform(0.5, 1.5)
        hsv = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = hsv[:, :, 2] * value
        hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        image = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)

    return image
