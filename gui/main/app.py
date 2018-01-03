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
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button

import sys
import threading
from functools import partial
from multiprocessing import Process

sys.path.append('../../')


# Get screen resolution
from screeninfo import get_monitors

monitor = get_monitors()[0]
screen_w, screen_h = monitor.width, monitor.height


# Detector objects

import detector
from detector.cropper import Cropper
from detector.localizer import Localizer
from detector.classifier import Classifier
from detector.detector import Detector
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
plotter = Plotter(num_classes=20, bgr=True)
detection_pipeline = DetectionPipeline(localizer, cropper, classifier)
detector = Detector(detection_pipeline)


# Get tensorflow graph
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

    def on_enter(self):
        self.ids.format_json.disabled = True
        self.ids.format_txt.disabled = True
        self.ids.format_csv.disabled = True
        self.ids.format_jpg.disabled = True

        self.manager.state_data.show_confidence_image = False
        self.manager.state_data.output_format_image = None


    def save_image_path(self):
        self.manager.state_data.image_path = self.ids.filechooser.selection[0]


    def save_confidence_choice(self, instance, value):
        self.manager.state_data.show_confidence_image = value


    def enable_format_checkbox(self, instance, value):
        if value:
            self.ids.format_json.disabled = False
            self.ids.format_txt.disabled = False
            self.ids.format_csv.disabled = False
            self.ids.format_jpg.disabled = False
        else:
            self.ids.format_json.disabled = True
            self.ids.format_txt.disabled = True
            self.ids.format_csv.disabled = True
            self.ids.format_jpg.disabled = True


    def check_image_options(self):
        error = False
        feed_path = None

        if self.ids.save_results_checkbox.active:
            if self.ids.format_json.active:
                self.manager.state_data.output_format_image = '.json'
            elif self.ids.format_txt.active:
                self.manager.state_data.output_format_image = '.txt'
            elif self.ids.format_csv.active:
                self.manager.state_data.output_format_image = '.csv'
            elif self.ids.format_jpg.active:
                self.manager.state_data.output_format_image = '.jpg'
            else:
                error = True
                error_msg = 'Seleccione un formato de salida'

        if error:
            box = BoxLayout(orientation='vertical')
            error_label = Label(text=error_msg, font_size=50)
            box.add_widget(error_label)
            close_button = Button(text='Ok', size_hint=(1, .3))
            box.add_widget(close_button)

            popup = Popup(title='Ha ocurrido un error', content=box, size_hint=(.5, .5))
            close_button.bind(on_release=popup.dismiss)
            popup.open()
        else:
            self.manager.current = 'image_result'
            self.manager.transition.direction = 'left'


    def on_leave(self):
        self.ids.filechooser.selection = ''
        self.ids.detect_button.disabled = True
        self.ids.confidence_checkbox.active = False
        self.ids.save_results_checkbox.active = False
        self.ids.format_json.active = False
        self.ids.format_txt.active = False
        self.ids.format_csv.active = False
        self.ids.format_jpg.active = False


class ImageResultScreen(Screen):

    def on_enter(self):
        image_path = self.manager.state_data.image_path
        show_confidence = self.manager.state_data.show_confidence_image

        _, file_extension = os.path.splitext(self.manager.state_data.image_path)

        output_format = self.manager.state_data.output_format_image

        if output_format is not None:
            output_filename = image_path[:-4] + '_results' + output_format
        else:
            output_filename = None

        if file_extension in ['.jpg', '.png', '.gif']:
            img = cv2.imread(image_path)

            # Detect objects in image
            detected_image, detections = detector.detect_image(img,
                                                               output=output_filename,
                                                               show_confidence=show_confidence,
                                                               return_image=True)

            flipped = cv2.flip(detected_image, 0)
            buf = flipped.tostring()
            image_texture = Texture.create(size=(detected_image.shape[1], detected_image.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

            self.ids.image_results.texture = image_texture


    def on_leave(self):
        self.ids.image_results.texture = Texture.create(size=(1,1))


class DetectFolderScreen(Screen):

    def is_dir(self, directory, filename):
        return os.path.isdir(os.path.join(directory, filename))


    def save_folder_path(self):
        self.manager.state_data.folder_path = self.ids.filechooser_folder.selection[0]


class FolderResultScreen(Screen):

    def detect_image(self, image_path):
        img = cv2.imread(image_path)
        detections = detector.detect_image(img)
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
        image_count = len(images)

        for image in images:
            global graph
            with graph.as_default():
                if self.on_screen:
                    self.detect_image(image)
                    self.ids.progress_bar.value += 1

                    percentage = int(self.ids.progress_bar.value / image_count * 100)
                    self.ids.percentage_text.text = str(percentage) + '%'
                    self.ids.progress_text.text = image
                else:
                    break


    def on_leave(self):
        self.on_screen = False
        self.ids.progress_bar.value =  0
        self.ids.progress_text.text = ''
        self.ids.percentage_text.text = ''


class DetectVideoScreen(Screen):

    def on_enter(self):
        self.manager.state_data.sound_notifications = False
        self.ids.switch_sound_notifications.active = False

    def switch_toggle(self, instance, value):
        self.manager.state_data.sound_notifications = value


    def save_video_path(self):
        self.manager.state_data.video_path = self.ids.filechooser.selection[0]


    def on_leave(self):
        self.ids.filechooser.selection = ''
        self.ids.detect_button.disabled = True


class VideoResultScreen(Screen):

    def on_enter(self):
        self.start_detection_thread()


    def start_detection_thread(self):
        video_path = self.manager.state_data.video_path

        thread = threading.Thread(target=self.detection_thread, args=(video_path,))
        thread.daemon = True
        thread.start()
        thread.join()


    def detection_thread(self, video_path):
        with graph.as_default():
            detector.detect_video_feed(video_path, show_output=True, sound_notifications=True)


class DetectWebcamScreen(Screen):

    def on_enter(self):
        self.manager.state_data.sound_notifications = False
        self.ids.switch_sound_notifications.active = False


    def switch_toggle(self, instance, value):
        self.manager.state_data.sound_notifications = value


    def check_input_feed(self):
        error = False
        feed_path = None

        if self.ids.checkbox_input_webcam.active:
            if self.ids.text_input_webcam.text is not '':
                feed_path = int(self.ids.text_input_webcam.text)
            else:
                error = True
                error_msg = 'Ingrese un ID de Webcam'
        elif self.ids.checkbox_input_url.active:
            if self.ids.text_input_url.text is not '':
                feed_path = self.ids.text_input_url.text
            else:
                error = True
                error_msg = 'Ingrese una URL de un feed de video'
        else:
            error = True
            error_msg = 'Seleccione una entrada de video'

        if error:
            box = BoxLayout(orientation='vertical')
            error_label = Label(text=error_msg, font_size=50)
            box.add_widget(error_label)
            close_button = Button(text='Ok', size_hint=(1, .3))
            box.add_widget(close_button)

            popup = Popup(title='Ha ocurrido un error', content=box, size_hint=(.5, .5))
            close_button.bind(on_release=popup.dismiss)
            popup.open()
        else:
            self.manager.state_data.feed_path = feed_path

            cap = cv2.VideoCapture(feed_path)
            ret, frame = cap.read()
            cap.release()

            if ret:
                self.manager.current = 'webcam_result'
                self.manager.transition.direction = 'left'
            else:
                box = BoxLayout(orientation='vertical')
                error_label = Label(text='No se pudo leer del feed de video', font_size=50)
                box.add_widget(error_label)
                close_button = Button(text='Ok', size_hint=(1, .3))
                box.add_widget(close_button)

                popup = Popup(title='Ha ocurrido un error', content=box, size_hint=(.5, .5))
                close_button.bind(on_release=popup.dismiss)
                popup.open()



    def on_leave(self):
        self.ids.checkbox_input_webcam.active = False
        self.ids.text_input_webcam.text = ''

        self.ids.checkbox_input_url.active = False
        self.ids.text_input_url.text = ''


class WebcamResultScreen(Screen):

    def on_enter(self):
        self.start_detection_thread()


    def start_detection_thread(self):
        thread = threading.Thread(target=self.detection_thread)
        thread.daemon = True
        thread.start()
        thread.join()


    def detection_thread(self):
        feed_path = self.manager.state_data.feed_path
        sound_notifications = self.manager.state_data.sound_notifications

        with graph.as_default():
            detector.detect_video_feed(feed_path,
                                       show_output=True,
                                       sound_notifications=sound_notifications)


# Define screen manager
class ScreenManagement(ScreenManager):
    state_data = ObjectProperty(DataManager())


# Load .kv file
ui = Builder.load_file('detector.kv')


# DetectorApp Class
class DetectorApp(App):
    title = 'Detector de objectos'
    icon = '/home/arian/stop.png'


    def build(self):
        return ui


# Create and run app
#Window.fullscreen = 'auto'
Window.size = (screen_w, screen_h)
app = DetectorApp()
app.run()
