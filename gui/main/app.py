import os
import cv2
import glob

from kivy.app import App
from kivy.lang import Builder
from kivy.config import Config
from kivy.core.window import Window
from kivy.event import EventDispatcher
from kivy.graphics.texture import Texture
from kivy.properties import StringProperty
from kivy.properties import ObjectProperty
from kivy.uix.screenmanager import Screen, ScreenManager

import sys
import threading

sys.path.append('../../')


# Decetor objects

from detector.cropper import Cropper
from detector.localizer import Localizer
from detector.classifier import Classifier
from detector.detection_pipeline import DetectionPipeline
from detector.plotter import Plotter


# Create objects
cropper = Cropper(0.25, force_square=True)
localizer = Localizer(cfg_path='../../data/yolo/full/trafficsigns.cfg',
                      weights_path='../../data/yolo/full/trafficsigns.weights',
                      threshold=0.1)
classifier = Classifier(model_path='../../data/classifier/trafficsigns.json',
                        weights_path='../../data/classifier/trafficsigns.h5',
                        labels_path='../../data/classifier/classes.txt',
                        threshold=0.9)
detector = DetectionPipeline(localizer, cropper, classifier)
plotter = Plotter(num_classes=20, bgr=True)

import tensorflow as tf

graph = tf.get_default_graph()


# Data manager
class DataManager(EventDispatcher):
    image_path = StringProperty()
    folder_path = StringProperty()


# Define app screens
class MainScreen(Screen):
    pass


class DetectImageScreen(Screen):

    def save_image_path(self):
        self.manager.state_data.image_path = self.ids.filechooser.selection[0]


class ImageResultScreen(Screen):

    def on_enter(self):
        # Show error if selected
        image_path = self.manager.state_data.image_path
        _, file_extension = os.path.splitext(self.manager.state_data.image_path)

        if file_extension in ['.jpg', '.png', '.gif']:
            img = cv2.imread(image_path)

            # Detect objects in image
            detections = detector.detect_objects_in_image(img)
            img = plotter.plot_detections(img,
                                          detections,
                                          draw_confidence=False)

            flipped = cv2.flip(img, 0)
            buf = flipped.tostring()
            image_texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

            self.ids.image_results.texture = image_texture


    def remove_picture(self):
        self.ids.image_results.texture = Texture.create(size=(1,1))


class DetectFolderScreen(Screen):

    def is_dir(self, directory, filename):
        return os.path.isdir(os.path.join(directory, filename))


    def save_folder_path(self):
        self.manager.state_data.folder_path = self.ids.filechooser_folder.selection[0]


class FolderResultScreen(Screen):

    def detect_image(self, image_path):
        img = cv2.imread(image_path)
        detections = detector.detect_objects_in_image(img)
        img = plotter.plot_detections(img,
                                      detections,
                                      draw_confidence=False)

        flipped = cv2.flip(img, 0)
        buf = flipped.tostring()
        image_texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

        image_filename = os.path.basename(image_path)
        output_path = os.path.join('/home/arian/Output', image_filename)
        cv2.imwrite(output_path, img)


    def on_enter(self):
        self.on_screen = True

        folder_path = self.manager.state_data.folder_path

        images = []
        for extension in ('*.gif', '*.png', '*.jpg', '*.JPEG'):
            images.extend(glob.glob(os.path.join(folder_path, extension)))

        images_count = len(images)
        self.ids.progress_bar.max = 0
        self.ids.progress_bar.max = images_count

        self.start_detection_thread(images)


    def start_detection_thread(self, images):
        threading.Thread(target=self.detection_thread, args=(images,)).start()


    def detection_thread(self, images):
        for image in images:
            global graph
            with graph.as_default():
                if self.on_screen:
                    self.detect_image(image)
                    self.ids.progress_bar.value += 1
                else:
                    break

    def on_leave(self):
        self.on_screen = False
        self.ids.progress_bar.value =  0



# Define screen manager
class ScreenManagement(ScreenManager):
    state_data = ObjectProperty(DataManager())


# Load .kv file
ui = Builder.load_file('detector.kv')


# DetectorApp Class
class DetectorApp(App):

    def build(self):
        Config.set('graphics', 'fullscreen', 'auto')
        return ui


# Create and run app
Window.fullscreen = 'auto'
app = DetectorApp()
app.run()
