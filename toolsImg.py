import os
from PIL import Image
import random
import cv2 
from mtcnn import MTCNN
import numpy as np

detector = MTCNN()

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
 
def resize_image(image, width):
    aspect_ratio = width / image.shape[1]
    height = int(image.shape[0] * aspect_ratio)
    return cv2.resize(image, (width, height))

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

def detect_and_draw_faces(image):

    img_cv2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if not isinstance(image, np.ndarray) else image

    result = detector.detect_faces(img_cv2)

    num_faces = len(result)
    for face in result:
        x1, y1, width, height = face['box']
        x1, y1 = abs(int(x1)), abs(int(y1))
        x2, y2 = x1 + int(width), y1 + int(height)
        img_cv2 = cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 0, 255), 5)
    
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    print("Number of faces detected: ", num_faces)
    return img_rgb

def main():
    folder_path = "D:\Project\Heineken\FULL [Heineken Vietnam] Developer Resources"
    img_list = os.listdir(folder_path)
    
    while True:
        print("Enter the index of the image you want to detect (0 to {}), or type 'exit' to quit:".format(len(img_list)-1))
        try:
            index = input("Index: ")
            if index.lower() == 'exit':
                break
            index = int(index)
            if index < 0 or index >= len(img_list):
                print("Invalid index. Please enter a valid index.")
                continue

            img_path = os.path.join(folder_path, img_list[index])
            image = cv2.imread(img_path)

            if image is None:
                print(f"Error: Image {img_path} not found or could not be loaded.")
                continue

            detected_img = detect_and_draw_faces(image)
            resize = resize_image(detected_img, 800)
            cv2.imshow('Detected Faces', resize)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except ValueError:
            print("Invalid input. Please enter a valid index.")