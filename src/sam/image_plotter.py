#!/usr/bin/env python
"""
pltで描画するモジュール
"""
import matplotlib.pyplot as plt


class ImagePlotter:
    """描画するためのクラス"""

    def __init__(self, size=(20, 20), axis="on") -> None:
        self.new_figure(size, axis)
        self.plot_area = plt.gca()

    def set_image(self, img):
        """imageをFigureにセット"""
        self.plot_area.imshow(img)

    @staticmethod
    def show():
        """pltにセットされているものをGUIで表示"""
        plt.show()

    @staticmethod
    def save(file_path):
        """pltにセットされているfigureを保存"""
        if "." not in file_path:
            file_path += ".png"
        print("save_path: " + file_path)
        plt.savefig(file_path)

    @staticmethod
    def new_figure(size=(20, 20), axis="on"):
        """figureを作成"""
        plt.figure(figsize=size)
        plt.axis(axis)

    def set_anns(self, anns):
        """透過マスク情報を表示"""
        if len(anns) == 0:
            return
        self.plot_area.set_autoscale_on(False)
        for ann in anns:
            self.plot_area.imshow(ann)
