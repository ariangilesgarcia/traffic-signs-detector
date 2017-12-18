import kivy

from kivy.app import App
from kivy.lang import Builder
from kivy.config import Config
from kivy.core.window import Window
from kivy.uix.screenmanager import Screen, ScreenManager


# Define app screens
class MainScreen(Screen):
    pass


class DetectImageScreen(Screen):
    pass


# Define screen manager
class ScreenManagement(ScreenManager):
    pass


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
