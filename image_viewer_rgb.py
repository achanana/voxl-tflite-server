""" Copyright (c) 2015-2016 Qualcomm Technologies, Inc.  All Rights Reserved.
    Qualcomm Technologies Proprietary and Confidential.

    Copyright (c) 2020- Modified by ModalAI to support RGB streaming
"""

import sys
import os
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import struct
import socket
import time
import argparse


class CameraDisplay(QLabel):
    def __init__(self, parent = None):
        super(CameraDisplay, self).__init__(parent)

    def update_frame(self, image):
        self.setPixmap(QPixmap.fromImage(image))


class ImageViewerGui(QWidget):
    camera_window_signal = pyqtSignal(QImage)

    def __init__(self, parent = None):
        super(ImageViewerGui, self).__init__(parent)

        # Image dislay
        self.camera_window = CameraDisplay()
        self.camera_window_signal.connect(self.camera_window.update_frame)
        self.img_format = QImage.Format_RGB888

        # Save button
        save_btn = QPushButton("Save image")
        save_btn.clicked.connect(lambda: self.send_command(100))

        # Undistort button
        undistort_btn = QPushButton("Undistort")
        undistort_btn.clicked.connect(lambda: self.send_command(90))
        undistort_down_btn = QPushButton("<--")
        undistort_down_btn.clicked.connect(lambda: self.send_command(89))
        undistort_up_btn = QPushButton("-->")
        undistort_up_btn.clicked.connect(lambda: self.send_command(91))

        # Gain and exposure buttons
        gain_label = QLabel("Gain")
        gain_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        gain_down_btn = QPushButton("<--")
        gain_down_btn.clicked.connect(lambda: self.send_command(189))
        gain_up_btn = QPushButton("-->")
        gain_up_btn.clicked.connect(lambda: self.send_command(190))
        exposure_label = QLabel("Exposure")
        exposure_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        exposure_down_btn = QPushButton("<--")
        exposure_down_btn.clicked.connect(lambda: self.send_command(191))
        exposure_up_btn = QPushButton("-->")
        exposure_up_btn.clicked.connect(lambda: self.send_command(192))

        # Set window layout
        grid = QGridLayout()
        grid.setSpacing(10)

        grid.addWidget(self.camera_window, 0, 9, 5, 5)
        grid.addWidget(save_btn, 0, 0, 1, 5)
        grid.addWidget(undistort_btn, 1, 1, 1, 3)
        grid.addWidget(undistort_down_btn, 1, 0, 1, 1)
        grid.addWidget(undistort_up_btn, 1, 4, 1, 1)
        grid.addWidget(gain_label, 2, 1, 1, 3)
        grid.addWidget(gain_down_btn, 2, 0, 1, 1)
        grid.addWidget(gain_up_btn, 2, 4, 1, 1)
        grid.addWidget(exposure_label, 3, 1, 1, 3)
        grid.addWidget(exposure_down_btn, 3, 0, 1, 1)
        grid.addWidget(exposure_up_btn, 3, 4, 1, 1)
        self.setLayout(grid)

        self.resize(500, 300)
        self.setWindowTitle("Image Viewer GUI")
        self.show()

        self.stream_connected = False

        # Create a dummy image
        img = QImage(300, 300, self.img_format)
        img.fill(qRgb(0,255,0))
        self.camera_window_signal.emit(img)

    def wait_for_connection(self):

        # Socket stuff
        try:
	    print "Trying to establish connection", self.ipaddr
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.ipaddr, 5556))
            self._packet_buf = []
            print "Connection initialized to", self.ipaddr

            self.stream_connected = True
            self.initialize_stream()

        except:
            QTimer.singleShot(100, self.wait_for_connection)


    def unpack_header(self):

        hdr = ''.join(self._packet_buf[:24])
        self.header = {}
        self.header['msg_id'] = struct.unpack('B', hdr[0])[0]
        self.header['flag'] = struct.unpack('B', hdr[1])[0]
        self.header['frame_id'] = struct.unpack('i', hdr[2:6])[0]
        self.header['timestamp_ns'] = struct.unpack('q', hdr[6:14])[0]
        self.header['num_cols'] = struct.unpack('H', hdr[14:16])[0]
        self.header['num_rows'] = struct.unpack('H', hdr[16:18])[0]
        self.header['opts'] = struct.unpack('B'*4, hdr[18:22])
        self.header['checksum'] = struct.unpack('H', hdr[22:24])[0]


    def initialize_stream(self):

        while len(self._packet_buf) < 24:
            data = self.sock.recv(24)
            self._packet_buf.extend(data)

        self.unpack_header()
        self.packet_size = 24 + self.header['num_cols'] * self.header['num_rows'] * 3

        if (self.header['msg_id'] == 2):
            self.resize(self.header['num_cols'], self.header['num_rows'])
        elif (self.header['msg_id'] == 20) or (self.header['msg_id'] == 21):
            self.resize(2*self.header['num_cols'], self.header['num_rows'])


    def send_command(self, message):
        self.sock.send(struct.pack('B', message))


    def handle_save_btn(self):
        self.send_command_signal.emit(100)


    def run(self):
        try:
            # Do some stuff here
            # This is the image receive loop
            # Continue to read in images and display until user quits

            if (not self.stream_connected):
               raise UserWarning("stream not connected")

            while len(self._packet_buf) < self.packet_size:
                data = self.sock.recv(self.packet_size)
                self._packet_buf.extend(data)

            self.unpack_header()

            # Monocular image
            if (self.header['msg_id'] == 2):
                frame = ''.join(self._packet_buf[24:self.packet_size])
                img = QImage(frame, self.header['num_cols'], self.header['num_rows'], self.img_format)
                self.camera_window_signal.emit(img)

            # Stereo image
            elif (self.header['msg_id'] == 20):
                self.left_frame = ''.join(self._packet_buf[24:self.packet_size])
                self.left_time = self.header['timestamp_ns']

            elif (self.header['msg_id'] == 21):
                self.right_frame = ''.join(self._packet_buf[24:self.packet_size])
                self.right_time = self.header['timestamp_ns']

                if (self.left_time == self.right_time):
                    frame = ''
                    start = 0
                    end = self.header['num_cols']
                    for r in xrange(self.header['num_rows']):
                        lr = self.left_frame[start:end]
                        rr = self.right_frame[start:end]
                        frame = ''.join([frame, lr, rr])
                        start = end
                        end += self.header['num_cols']

                    img = QImage(frame, 2*self.header['num_cols'], \
                            self.header['num_rows'], \
                            2*self.header['num_cols'], self.img_format)
                    img.setColorTable(self.img_color_table)
                    self.camera_window_signal.emit(img)

            # Clear the packet buffer to read the next image
            while (len(self._packet_buf) >= self.packet_size):
                self._packet_buf = self._packet_buf[self.packet_size:]

        except UserWarning, e:
            pass

        finally:
            QTimer.singleShot(10, self.run)


def main():
    # Do stuff here
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--ipaddress', type=str, help="IP address of the vehicle", required=True)
    args = parser.parse_args()

    app = QApplication(sys.argv)
    ivg = ImageViewerGui()
    ivg.ipaddr = args.ipaddress
    ivg.wait_for_connection()
    ivg.run()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
