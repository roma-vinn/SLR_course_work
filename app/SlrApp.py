from kivy.app import App
from kivy.uix.screenmanager import ScreenManager
from modules import real_time, model, predict, prepare_video_capture, DICT
from kivy.lang.builder import Builder
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import Screen
from kivy.uix.label import Label
import numpy as np
import struct

Builder.load_file('main.kv')  # load .kv markup file


class FileRecognitionScreen(Screen):
    """
    implementation screen responsible
    for recognition by a file selected in advance
    """

    def upload(self):
        """
        uploading video file from FileChooser widget and
        predicting sign number
        :return: None
        """
        file_path = self.file_chooser.selection[0]  # get path from widget

        try:
            cur_sign = prepare_video_capture(file_path)  # get an array of frames
            cur_sign = np.expand_dims(cur_sign.reshape(*cur_sign.shape, 1), axis=0)
            res = predict(model, cur_sign)  # get prediction
            self.predict_label.text = f"predicted sign: {DICT[res]}"  # output

        except Exception as e:
            #  if the selected file was in the wrong format
            popup = Popup(title='Error',
                          content=Label(text="invalid file"),
                          size_hint=(None, None), size=(300, 250))
            popup.open()
        # playback the video file
        self.video_playback.source = file_path
        self.video_playback.play = True


class StartScreen(Screen):
    """
    implementation of the main screen
    """
    pass


class TrainingScreen(Screen):
    """
    Implementation of training screen
    allowing user to learn new sign languages gestures
    """

    def train(self):
        """
        taking word from text input widget and showing corresponding video file
        :return: None
        """
        word = self.input.text
        word = word.lower()
        word = word.capitalize()
        ind = -1
        try:
            ind = DICT.index(word)
        except ValueError:
            pass
        if ind in range(10):
            self.video_train.source = f"data/{ind+1}.mp4"
            self.video_train.play = True
        else:
            #  generate the popup if the word not in database
            popup = Popup(title='Error',
                          content=Label(text="no such word\nin database"),
                          size_hint=(None, None), size=(300, 250))
            popup.open()


class RealTimeRecognitionScreen(Screen):
    """
    implementation screen
    real-time sign recognition
    """

    def __init__(self, **kw):
        """
        constructor
        :param kw: **kwargs
        """
        super().__init__(**kw)  # super init call
        self._h1 = None  # h (0 <=h <= 180) in hsv color representation (color for the one hand)
        self._h2 = None  # h (0 <=h <= 180) in hsv color representation (color for the other hand)

    def get_touch(self, pos):
        """
        finding the coordinates of the touch on the image to obtain the color of the pixel
        :param pos: touch position
        :return: None
        """
        if not self.camera.collide_point(*pos):  # if touch was not in image widget
            return False
        else:
            wx, wy = self.camera.pos  # coordinates of the lower left corner of the widget
            x, y = pos  # coordinates of the touch
            self.get_coord((x - wx, y - wy), self.camera.texture)  # position in widget coordinate system

    def get_color1(self):
        """
        selecting color for the one hand
        :return: None
        """
        h, s, v = self.col_picker.hsv  # get color in hsv format
        self.color1.background_color = self.col_picker.color  # repainting button
        self._h1 = h * 180  # saving to class field

    def get_color2(self):
        """
        selecting color for the other hand
        :return: None
        """
        h, s, v = self.col_picker.hsv  # get color in hsv format
        self.color2.background_color = self.col_picker.color  # repainting button
        self._h2 = h * 180  # saving to class field

    def reset(self):
        """
        reconfiguration selected colors
        :return: None
        """
        #  setting both colors as None
        self._h1 = None
        self._h2 = None
        #  repainting buttons
        self.color1.background_color = self.example.background_color
        self.color2.background_color = self.example.background_color

    def recognize(self):
        """
        starting real time recognition
        :return: None
        """
        real_time(self._h1, self._h2)

    def _get_color_value(self, pos, texture):
        """
        getting color in hsv format
        :param pos: coordinates of the pixel
        :param texture: texture (kivy class)
        :return: an array: [h, s, v] color representation
        """
        pixel = texture.get_region(pos[0], pos[1], 1, 1)  # get region object contains single pixel
        bp = pixel.pixels  # get pixel
        data = struct.unpack('4B', bp)  # unpack binary string
        return [x / 255.0 for x in data]  # percentage conversion

    def get_coord(self, pos, texture):
        """
        repainting color picker widget as the pixel color
        :param pos: pixel coordinates
        :param texture: texture object
        :return: None
        """
        x, y = pos
        self.col_picker.color = self._get_color_value((x, y), texture)  # repainting


class SlrApp(App):
    """
    implementation of the main app
    """

    def build(self):
        """
        building application
        :return: screen manager - main widget
        """
        sm = ScreenManager()  # creating screen manager
        #  adding screens
        sm.add_widget(StartScreen(name="start_screen"))
        sm.add_widget(FileRecognitionScreen(name="file_screen"))
        sm.add_widget(RealTimeRecognitionScreen(name="real_time_screen"))
        sm.add_widget(TrainingScreen(name="training_screen"))
        return sm


if __name__ == '__main__':
    #  run for build the app
    SlrApp().run()
