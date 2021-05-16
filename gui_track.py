import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
import pydicom
import numpy as np
from detect_edge import image_segmentation, get_vessel_points, get_initial_path, get_candidate_point

x_neighbor = [1, 1, 0, -1, -1, -1, 0, 1]
y_neighbor = [0, 1, 1, 1, 0, -1, -1, -1]
dic_images = pydicom.read_file("00000000")
np_images = np.array(dic_images.pixel_array)
total_image_num, _, _ = np_images.shape


def my_message_box(message_str):
    alert = QMessageBox()
    alert.setText(message_str)
    alert.exec_()


class Form(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Image Segmentation'
        self.label = None
        self.mousePressEvent = self.getPos
        self.label1 = None

        self.x_start = 0
        self.y_start = 0
        self.x_end = 0
        self.y_end = 0
        self.start_selected = False
        self.end_selected = False
        self.skeleton_image = []
        self.filter_image = []
        self.start_template_image = []
        self.end_template_image = []
        self.process_image = None
        self.cur_image_num = 30
        self.process_index = 0

    def iniUI(self):
        self.setWindowTitle(self.title)

        self.label = QLabel(self)
        self.label1 = QLabel(self)

        for i in range(self.cur_image_num, total_image_num):
            skeleton_im, filter_im = image_segmentation(np_images[i])
            self.skeleton_image.append(skeleton_im)
            self.filter_image.append(filter_im)

        self.process_image = np.zeros(np_images[self.cur_image_num].shape, dtype=np.uint8)

        self.drawUI()

        btn_brow_1 = QPushButton('Reset Points', self)
        btn_brow_1.resize(btn_brow_1.sizeHint())
        btn_brow_1.move(540, 50)
        btn_brow_1.clicked.connect(self.ResetPoints)

        btn_brow_2 = QPushButton('Get Vessel Path', self)
        btn_brow_2.resize(btn_brow_2.sizeHint())
        btn_brow_2.move(535, 80)
        btn_brow_2.clicked.connect(self.GetVesselPath)

        btn_brow_3 = QPushButton('Next Image', self)
        btn_brow_3.resize(btn_brow_2.sizeHint())
        btn_brow_3.move(535, 110)
        btn_brow_3.clicked.connect(self.NextImageProcessing)

        btn_brow_4 = QPushButton('Prev Image', self)
        btn_brow_4.resize(btn_brow_2.sizeHint())
        btn_brow_4.move(535, 140)
        btn_brow_4.clicked.connect(self.PreviousImageProcessing)

        self.show()

    def drawUI(self):
        id_num = self.cur_image_num
        image1 = np.copy(np_images[id_num])
        image2 = np.copy(np_images[id_num])
        image1[self.skeleton_image[self.process_index] == 1] = 255
        image2[self.process_image == 1] = 255

        if self.start_selected:
            for i in range(9):
                image1[self.y_start][self.x_start + i - 4] = 255
                image2[self.y_start + i - 4][self.x_start] = 255
                image1[self.y_start + i - 4][self.x_start] = 255
                image2[self.y_start][self.x_start + i - 4] = 255

            if self.end_selected:
                for j in range(9):
                    image1[self.y_end][self.x_end + j - 4] = 255
                    image2[self.y_end + j - 4][self.x_end] = 255
                    image1[self.y_end + j - 4][self.x_end] = 255
                    image2[self.y_end][self.x_end + j - 4] = 255

        qimage = QImage(image1, image1.shape[1], image1.shape[0], QImage.Format_Grayscale8)
        pixmap = QPixmap(qimage)
        self.label.setGeometry(10, 10, pixmap.width(), pixmap.height())
        self.label.setPixmap(pixmap)

        qimage = QImage(image2, image2.shape[1], image2.shape[0], QImage.Format_Grayscale8)
        pixmap1 = QPixmap(qimage)
        self.label1.setGeometry(pixmap.width() + 120, 10, pixmap.width(), pixmap.height())
        self.label1.setPixmap(pixmap1)
        self.resize(pixmap.width() * 2 + 130, pixmap.height() + 20)

    def ResetPoints(self):
        self.x_start = 0
        self.y_start = 0
        self.x_end = 0
        self.y_end = 0
        self.start_selected = False
        self.end_selected = False
        self.process_image = np.zeros(self.skeleton_image[0].shape, dtype=np.uint8)
        self.start_template_image = []
        self.end_template_image = []
        self.drawUI()
        self.show()

    def get_vessel_path(self, x0, y0, x1, y1):
        self.skeleton_image[self.process_index][y0][x0] = 0
        final_path = get_initial_path(self.skeleton_image[self.process_index], x0, y0, x1, y1)
        self.skeleton_image[self.process_index][y0][x0] = 1

        if len(final_path) == 0:
            return False
        self.process_image = np.zeros(self.skeleton_image[self.process_index].shape, dtype=np.uint8)
        for x, y in final_path:
            self.process_image[y][x] = 1

        self.start_template_image = []
        self.end_template_image = []
        neighbor_pts = [[0, 0], [-16, 0], [16, 0], [0, -16], [0, 16]]
        height, width = self.skeleton_image[self.process_index].shape
        for i in range(5):
            y = max(16, min(height - 16, y0 + neighbor_pts[i][1]))
            x = max(16, min(width - 16, x0 + neighbor_pts[i][0]))
            image = self.filter_image[self.process_index][y - 16:y + 16, x - 16:x + 16]
            self.start_template_image.append(image)
            x = x1 + neighbor_pts[i][0]
            y = y1 + neighbor_pts[i][1]
            image = self.filter_image[self.process_index][y - 16:y + 16, x - 16:x + 16]
            self.end_template_image.append(image)

        return True

    def GetVesselPath(self):
        if not self.start_selected:
            my_message_box('Select Start Point!')
            return
        if not self.end_selected:
            my_message_box('Select End Point!')
            return

        res = self.get_vessel_path(self.x_start, self.y_start, self.x_end, self.y_end)
        if not res:
            my_message_box('Can not find the connection ridge!')
            return

        self.drawUI()
        self.show()

    def getPos(self, event):
        x = event.pos().x()
        y = event.pos().y()

        ENLARGE_LENGTH = 10
        if x < ENLARGE_LENGTH or y < ENLARGE_LENGTH or x >= self.label.width() + ENLARGE_LENGTH or \
                y >= self.label.height() + ENLARGE_LENGTH:
            return
        if self.end_selected:
            return

        min_r, x1, y1 = get_vessel_points(self.skeleton_image[self.process_index], x - ENLARGE_LENGTH,
                                         y - ENLARGE_LENGTH)
        if min_r == 500:
            my_message_box('Reselect Points!')
            return

        if not self.start_selected:
            self.x_start = x1
            self.y_start = y1
            self.start_selected = True
        else:
            if not self.end_selected:
                self.x_end = x1
                self.y_end = y1
                self.end_selected = True
        self.drawUI()
        self.show()

    def processing_image(self):
        self.process_image = np.zeros(self.skeleton_image[self.process_index].shape, dtype=np.uint8)
        x, y = get_candidate_point(self.filter_image[self.process_index], self.start_template_image,
                                   self.x_start, self.y_start)
        min_r, x1, y1 = get_vessel_points(self.skeleton_image[self.process_index], x, y)
        if min_r == 500:
            my_message_box('Not find the start point!')
            return

        self.x_start = x1
        self.y_start = y1
        self.start_selected = True

        x, y = get_candidate_point(self.filter_image[self.process_index], self.end_template_image,
                                   self.x_end, self.y_end)
        min_r, x1, y1 = get_vessel_points(self.skeleton_image[self.process_index], x, y)
        if min_r == 500:
            my_message_box('Not find the end point!')
            return

        self.x_end = x1
        self.y_end = y1
        self.end_selected = True

        res = self.get_vessel_path(self.x_start, self.y_start, self.x_end, self.y_end)
        if not res:
            my_message_box('Can not find the connection ridge!')

    def NextImageProcessing(self):
        if self.cur_image_num >= total_image_num - 1:
            my_message_box('There is not any image now!')
            return

        self.process_index += 1
        self.cur_image_num += 1
        if len(self.start_template_image) > 0:
            self.processing_image()
        else:
            self.start_selected = False
            self.end_selected = False

        self.drawUI()
        self.show()

    def PreviousImageProcessing(self):
        if self.process_index == 0:
            my_message_box('There is not any image now!')
            return

        self.process_index -= 1
        self.cur_image_num -= 1
        if len(self.start_template_image) > 0:
            self.processing_image()
        else:
            self.start_selected = False
            self.end_selected = False

        self.drawUI()
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Form()
    ex.iniUI()
    sys.exit(app.exec_())
