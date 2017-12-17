import kivy

from kivy.app import App
from kivy.config import Config
from kivy.uix.boxlayout import BoxLayout


class DetectorLayout(BoxLayout):
    pass


class DetectorApp(App):

    def build(self):
        Config.set('graphics', 'width', '1280')
        Config.set('graphics', 'height', '720')

        return DetectorLayout()


app = DetectorApp()
app.run()
