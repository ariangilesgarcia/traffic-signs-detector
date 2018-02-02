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

sys.path.append('../../')


# Get screen resolution
from screeninfo import get_monitors

monitor = get_monitors()[0]
screen_w, screen_h = monitor.width, monitor.height


# Detector objects

import detector
from detector.detector import create_detector_from_file

# Get tensorflow graph
import tensorflow as tf
graph = tf.get_default_graph()


# Data manager
class DataManager(EventDispatcher):
    pass


# Define app screens
class ConfigScreen(Screen):

    def save_config_path(self):
        self.manager.state_data.config_path = '../../cfg/config.json'


class LoadConfigScreen(Screen):

    def save_config_path(self):
        self.manager.state_data.config_path = self.ids.filechooser.selection[0]


    def on_leave(self):
        self.ids.filechooser.selection = ''
        self.ids.select_button.disabled = True


class CreateDetectorScreen(Screen):

    def on_enter(self):
        cfg_path = self.manager.state_data.config_path

        try:
            self.manager.state_data.detector = create_detector_from_file(cfg_path)
            self.manager.current = 'main'
            self.manager.transition.direction = 'left'
        except:
            print('!!! - Error creating detector from file!')

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
            detected_image, detections = self.manager.state_data.detector.detect_image(img,
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

    def on_enter(self):
        self.manager.state_data.output_format_image = None
        self.manager.state_data.output_folder_path = None


    def is_dir(self, directory, filename):
        return os.path.isdir(os.path.join(directory, filename))


    def enable_output_folder(self, instance, value):
        if value:
            self.ids.output_filechooser_folder.disabled = False
        else:
            self.ids.output_filechooser_folder.disabled = True


    def save_folder_path(self):
        self.manager.state_data.folder_path = self.ids.filechooser_folder.selection


    def save_output_path(self):
        self.manager.state_data.output_folder_path = self.ids.output_filechooser_folder.selection


    def check_folder_options(self):
        """
        root.save_folder_path()
        filechooser_folder.selection = ''
        app.root.current = 'folder_result'
        app.root.transition.direction = 'left'
        """

        error = False
        folder_path = None

        if self.ids.format_json.active:
            self.manager.state_data.output_format_folder = '.json'
        elif self.ids.format_txt.active:
            self.manager.state_data.output_format_folder = '.txt'
        elif self.ids.format_csv.active:
            self.manager.state_data.output_format_folder = '.csv'
        elif self.ids.format_jpg.active:
            self.manager.state_data.output_format_folder = '.jpg'
        else:
            error = True
            error_msg = 'Seleccione un formato de salida'


        if self.ids.output_folder_chechbox.active:
            if self.manager.state_data.output_folder_path is None:
                error = True
                error_msg = 'Seleccione una carpeta de salida'

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
            self.manager.current = 'folder_result'
            self.manager.transition.direction = 'left'


    def on_leave(self):
        self.ids.filechooser_folder.selection = ''
        self.ids.output_filechooser_folder.selection = ''

        self.ids.detect_button.disabled = True

        self.ids.format_json.active = True
        self.ids.format_txt.active = False
        self.ids.format_csv.active = False
        self.ids.format_jpg.active = False


class FolderResultScreen(Screen):

    def on_enter(self):
        self.on_screen = True

        folder_path = self.manager.state_data.folder_path[0]

        images = []
        for extension in ('*.gif', '*.png', '*.jpg'):
            images.extend(glob.glob(os.path.join(folder_path, extension)))

        images_count = len(images)

        self.ids.progress_bar.max = 0
        self.ids.progress_bar.max = images_count

        self.start_detection_thread(images)


    def start_detection_thread(self, images):
        threading.Thread(target=self.detection_thread, args=(images,)).start()


    def detection_thread(self, images):
        if self.manager.state_data.output_folder_path:
            output_path = self.manager.state_data.output_folder_path[0]
        else:
            output_path = self.manager.state_data.folder_path[0]

        extension = self.manager.state_data.output_format_folder

        image_count = len(images)

        for image in images:
            global graph
            with graph.as_default():
                if self.on_screen:
                    image_output_basename = os.path.basename(image)[:-4] + '_output' + extension
                    output_filename = os.path.join(output_path, image_output_basename)

                    img = cv2.imread(image)
                    self.manager.state_data.detector.detect_image(img, output=output_filename)

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

        self.ids.format_csv.active = False
        self.ids.format_avi.active = False

        self.manager.state_data.csv_output = False
        self.manager.state_data.avi_output = False

        self.manager.state_data.show_confidence_video = False


    def switch_toggle(self, instance, value):
        self.manager.state_data.sound_notifications = value


    def save_video_path(self):
        self.manager.state_data.video_path = self.ids.filechooser.selection[0]


    def save_confidence_choice(self, instance, value):
        self.manager.state_data.show_confidence_video = value

    def check_video_options(self):
        error = False
        feed_path = None

        if self.ids.format_csv.active:
            self.manager.state_data.csv_output = True

        if self.ids.format_avi.active:
            self.manager.state_data.avi_output = True

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
            self.manager.current = 'video_result'
            self.manager.transition.direction = 'left'


    def on_leave(self):
        self.ids.filechooser.selection = ''
        self.ids.detect_button.disabled = True
        self.ids.confidence_checkbox.active = False
        self.ids.format_csv.active = False
        self.ids.format_avi.active = False


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
        show_confidence = self.manager.state_data.show_confidence_video
        sound_notifications = self.manager.state_data.sound_notifications

        save_csv = self.manager.state_data.csv_output
        if save_csv:
            if type(video_path) == int:
                csv_path = './webcam_results.csv'
            else:
                csv_path = video_path[:-4] + '_results' + '.csv'
        else:
            csv_path = None

        save_avi = self.manager.state_data.avi_output
        if save_avi:
            if type(video_path) == int:
                avi_path = './webcam_results.avi'
            else:
                avi_path = video_path[:-4] + '_results' + '.avi'
        else:
            avi_path = None

        with graph.as_default():
            self.manager.state_data.detector.detect_video_feed(video_path,
                                       show_output=True,
                                       output=avi_path,
                                       output_csv=csv_path,
                                       sound_notifications=sound_notifications,
                                       show_confidence=show_confidence)


class DetectWebcamScreen(Screen):

    def on_enter(self):
        self.manager.state_data.sound_notifications = False
        self.ids.switch_sound_notifications.active = False
        self.ids.format_csv.active = False
        self.ids.format_avi.active = False
        self.ids.confidence_checkbox = False
        self.manager.state_data.csv_output = False
        self.manager.state_data.avi_output = False
        self.manager.state_data.show_confidence_webcam = False


    def save_confidence_choice(self, instance, value):
        self.manager.state_data.show_confidence_webcam = value


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

        if self.ids.format_csv.active:
            self.manager.state_data.csv_output = True

        if self.ids.format_avi.active:
            self.manager.state_data.avi_output = True

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
        feed_path = self.manager.state_data.feed_path

        thread = threading.Thread(target=self.detection_thread, args=(feed_path,))
        thread.daemon = True
        thread.start()
        thread.join()


    def detection_thread(self, feed_path):
        show_confidence = self.manager.state_data.show_confidence_webcam
        sound_notifications = self.manager.state_data.sound_notifications

        save_csv = self.manager.state_data.csv_output

        if save_csv:
            csv_path = './webcam_results.csv'
        else:
            csv_path = None

        save_avi = self.manager.state_data.avi_output

        if save_avi:
            avi_path = './webcam_results.avi'
        else:
            avi_path = None

        with graph.as_default():
            self.manager.state_data.detector.detect_video_feed(feed_path,
                                       show_output=True,
                                       output=avi_path,
                                       output_csv=csv_path,
                                       sound_notifications=sound_notifications,
                                       show_confidence=show_confidence)


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
