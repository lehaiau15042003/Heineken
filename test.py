import cv2
import os
import numpy as np

def img(folder_path,num_images_to_augment):
    file_list = os.listdir(folder_path)
    output_folder = r'E:\Heineken\Heineken\image_Heineken_augmented'
    os.makedirs(output_folder, exist_ok=True)
    angles = [i * 30 for i in range(12)]
    count = 0
    for filename in file_list:
        if count >= num_images_to_augment:
            break
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        if image is not None:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            for angle in angles:
                (h, w) = gray_image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_image = cv2.warpAffine(gray_image, M, (w, h))
                if np.random.random() < 0.5:
                    rotated_image = cv2.flip(rotated_image, 1)
                brightness = np.random.uniform(0.5, 1.5)
                brightened_image = cv2.convertScaleAbs(rotated_image, alpha=brightness, beta=0)
                output_path = os.path.join(output_folder, f'augmented_{filename}_{angle}.jpg')
                cv2.imwrite(output_path, brightened_image)
                print(f"Đã tăng cường và lưu ảnh: {filename} -> {output_path}, kích thước mới: {brightened_image.shape}")
                count += 1
                if count >= num_images_to_augment:
                    break
        else:
            print(f"Không thể đọc ảnh: {filename}")

    print(f"Đã tăng cường và lưu tổng cộng {count} ảnh từ thư mục {folder_path} vào {output_folder}")

img('E:\Heineken\Heineken\heineken-image-analysis\[Heineken Vietnam] Developer Resources',1000)