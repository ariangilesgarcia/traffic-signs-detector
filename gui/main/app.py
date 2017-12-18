import cv2

from kivy.app import App
from kivy.lang import Builder
from kivy.config import Config
from kivy.core.window import Window
from kivy.event import EventDispatcher
from kivy.graphics.texture import Texture
from kivy.properties import StringProperty
from kivy.properties import ObjectProperty
from kivy.uix.screenmanager import Screen, ScreenManager

# Data manager
class DataManager(EventDispatcher):
    image_path = StringProperty()


# Define app screens
class MainScreen(Screen):
    pass


class DetectImageScreen(Screen):

    def save_image_path(self):
        self.manager.state_data.image_path = self.ids.filechooser.selection[0]


class ImageResultScreen(Screen):

    def on_enter(self):
        img = cv2.imread(self.manager.state_data.image_path)
        buf1 = cv2.flip(img, 0)
        buf = buf1.tostring()
        image_texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

        self.ids.image_results.texture = image_texture

    def remove_picture(self):
        self.ids.image_results.texture = Texture.create(size=(1,1))


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
