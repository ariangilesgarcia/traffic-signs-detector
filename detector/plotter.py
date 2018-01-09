import re
import cv2
import randomcolor


class Plotter:

    def __init__(self, num_classes, bgr=False):
        # Generate color for each class
        rand_color = randomcolor.RandomColor()
        colors = rand_color.generate(luminosity='bright', count=num_classes, format_='rgb')

        # Class map dictionary
        self.__class_map = {}
        class_id = 0

        # RGB string to R, G, B values
        p = re.compile(r'rgb\((\d{1,3}), (\d{1,3}), (\d{1,3})\)')

        # Populate dictionary
        for color in colors:
            matches = p.findall(color)
            r, g, b = matches[0]

            if bgr:
                class_color = (int(b), int(g), int(r))
            else:
                class_color = (int(r), int(g), int(b))

            self.__class_map[class_id] = {}
            self.__class_map[class_id]['color'] = class_color

            class_id += 1


    def plot_detections(self, image, detections, output=None, draw_confidence=False):
         for detection in detections:
            x1, y1, x2, y2 = detection['coordinates']
            label_id = detection['class_id']
            label = detection['label']

            # Get color for bbox
            color = self.__class_map[label_id]['color']
            type(color)

            # Line thickness
            bbox_thickness = 10

            # Draw bounding box
            image = cv2.rectangle(image, (x2, y2), (x1, y1), color, bbox_thickness)

            # Font and text configuration
            font = cv2.FONT_HERSHEY_DUPLEX
            scale = 1
            thickness = 2

            # Get shape of image
            img_height, img_width, _ = image.shape

            # Label text variables
            if draw_confidence:
                confidence = '{0:.0f}%'.format(detection['confidence']*100)
                label = '[{}] {}'.format(confidence, label)

            text = cv2.getTextSize(label, font, scale, thickness)
            text_width, text_height = text[0]

            x_text = x1
            y_text = y1 - 20

            x1_bg = x_text
            y1_bg = y1
            x2_bg = x_text + text_width
            y2_bg = y1 - 30 - text_height

            if x_text + text_width > img_width:
                x_text = img_width - text_width

            if y_text - text_height < 0:
                y_text = text_height + (y2-y1) + 20
                y1_bg = y2
                y2_bg = y2 + 20 + text_height

            # Draw rectangle as background text
            image = cv2.rectangle(image,
                                  (x1_bg, y1_bg),
                                  (x2_bg, y2_bg),
                                  color,
                                  bbox_thickness)

            image = cv2.rectangle(image,
                                  (x1_bg, y1_bg),
                                  (x2_bg, y2_bg),
                                  color,
                                  -1)

            # Draw text
            image = cv2.putText(image,
                                label,
                                (x_text, y_text),
                                cv2.FONT_HERSHEY_DUPLEX,
                                scale,
                                (255, 255, 255),
                                thickness)

            # Save image if the ouput paramter is set
            if output:
                cv2.imwrite(output, image)

         return image
