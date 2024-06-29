import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from PIL import Image

checkpoint_path = 'models/Heineken/models/official/vision/efficientdet/model_dir/efficientdet_d0_coco17_tpu-32/checkpoint/ckpt-0'
labels_path = 'models/Heineken/models/official/vision/efficientdet/model_dir/label_map.pbtxt'

num_classes = 1

label_map = label_map_util.load_labelmap(labels_path)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=num_classes,
    use_display_name=True
)
category_index = label_map_util.create_category_index(categories)


detection_model = tf.saved_model.load(checkpoint_path)

def detect_and_count(img_path):
    img_np = np.array(Image.open(img_path))
    input_tensor = tf.convert_to_tensor(img_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detection_model(input_tensor)

    num_detection = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detection].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detection
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = img_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False
    )

    num_customers = np.sum(detections['detection_classes'] == 1)
    return image_np_with_detections, num_customers

img_path = 'BZ1A0672.jpg'
output_image, num_customers = detect_and_count(img_path)

Image.fromarray(output_image).save('output_img.jpg')
print(f"Number of customers: {num_customers}")