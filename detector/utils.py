"""

Detection default format:
-------------------------

{
    'coordinates': [200, 100, 300, 200],
    'confidence': 0.998,
    'class_id': 16,
    'label': 'contramano'
}

"""


def convert_detection_to_csv(detection, include_label=False, include_confidence=False):
    label = detection['label']
    class_id = detection['class_id']
    confidence = detection['confidence']
    coordinates = detection['coordinates']

    detection_items = []

    detection_items.append(class_id)
    if include_label:
        detection_items.append(label)
    if include_confidence:
        detection_items.append(confidence)

    detection_items.extend(coordinates)

    detection_items = [str(x) for x in detection_items]

    csv_detection = ', '.join(detection_items)

    return csv_detection


def convert_detection_to_yolo(detection, image_size):
    coordinates = detection['coordinates']
    class_id = detection['class_id']

    relative_coordinates = convert_coordinates_to_relative(coordinates, image_size)
    label_items = [str(x) for x in relative_coordinates]
    label_items.insert(0, str(class_id))
    yolo_detection = ' '.join(label_items)

    return yolo_detection


def convert_coordinates_to_relative(box, size):
    dw = 1./size[0]
    dh = 1./size[1]

    x1 = (box[0] + box[2])/2.0
    y1 = (box[1] + box[3])/2.0

    w1 = box[2] - box[0]
    h1 = box[3] - box[1]

    x1 *= dw
    w1 *= dw
    y1 *= dh
    h1 *= dh

    return x1, y1, w1, h1
