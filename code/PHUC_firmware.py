import sys
import argparse
import select
import time
import numpy as np
#import cv2
from picamera2 import Picamera2
import socket
import pickle
#import adafruit_servokit
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
import board
import RPi.GPIO as GPIO


GPIO.setmode(GPIO.BCM)

FIRMWARE_VERSION = 1.0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_led_red', type=int, default=0,help='PCA9685 chan for left red LED')
    parser.add_argument('--left_led_green', type=int, default=1, help='PCA9685 chan for left green LED')
    parser.add_argument('--left_led_blue', type=int, default=2, help='PCA9685 chan for left blue LED')
    parser.add_argument('--top_led_red', type=int, default=5, help='PCA9685 chan for top red LED')
    parser.add_argument('--top_led_green', type=int, default=6, help='PCA9685 chan for top green LED')
    parser.add_argument('--top_led_blue', type=int, default=7, help='PCA9685 chan for top blue LED')
    parser.add_argument('--front_led_red', type=int, default=9, help='PCA9685 chan for front red LED')
    parser.add_argument('--front_led_green', type=int, default=10, help='PCA9685 chan for front green LED')
    parser.add_argument('--front_led_blue', type=int, default=11, help='PCA9685 chan for front blue LED')
    parser.add_argument('--right_led_red', type=int, default=13, help='PCA9685 chan for right red LED')
    parser.add_argument('--right_led_green', type=int, default=14, help='PCA9685 chan for right green LED')
    parser.add_argument('--right_led_blue', type=int, default=15, help='PCA9685 chan for right blue LED')
    parser.add_argument('--left_wheel', type=int, default=3, help='PCA9685 chan for left wheel servo')
    parser.add_argument('--right_wheel', type=int, default=4, help='PCA9685 chan for right wheel servo')
    parser.add_argument('--left_wheel_neutral', type=int, default=85, help='left wheel servo neutral setting')
    parser.add_argument('--right_wheel_neutral', type=int, default=90, help='right wheel servo neutral setting')
    parser.add_argument('--left_wheel_acceleration', type=int, default=10, help='left wheel accleration setting')
    parser.add_argument('--right_wheel_acceleration', type=int, default=10, help='right wheel acceleration setting')
    parser.add_argument('--left_upper_quadrant_trigger', type=int, default=20, help='RPI GPIO pin for LUQ TRIG')
    parser.add_argument('--left_upper_quadrant_echo', type=int, default=21, help='RPI GPIO pin for LUQ TRIG')
    parser.add_argument('--right_upper_quadrant_trigger', type=int, default=23, help='RPI GPIO pin for RUQ TRIG')
    parser.add_argument('--right_upper_quadrant_echo', type=int, default=24, help='RPI GPIO pin for RUQ TRIG')
    parser.add_argument('--left_lower_quadrant_trigger', type=int, default=19, help='RPI GPIO pin for LLQ TRIG')
    parser.add_argument('--left_lower_quadrant_echo', type=int, default=26, help='RPI GPIO pin for LLQ TRIG')
    parser.add_argument('--right_lower_quadrant_trigger', type=int, default=27, help='RPI GPIO pin for RLQ TRIG')
    parser.add_argument('--right_lower_quadrant_echo', type=int, default=22, help='RPI GPIO pin for RLQ TRIG')
    parser.add_argument('--camera_width', type=int, default=160, help='Picamera2 image width')
    parser.add_argument('--camera_height', type=int, default=120, help='Picamera2 image height')
    parser.add_argument('--camera_x_location', type=int, default=0, help='x location of sub image image')
    parser.add_argument('--camera_y_location', type=int, default=0, help='y location of sub image')
    parser.add_argument('--camera_resolution', type=int, default=1, help='determine pixel skipping in image')
    parser.add_argument('--camera_frames_per_second', type=int, default=24, help='camera fps')
    parser.add_argument('--port', type=int, default=2230, help='socket connection')
    args = parser.parse_args()
    return args

class PHUC_firmware:
    def __init__(self):
        args = parse_args()

        # Socket connection for camera image array
        self.port = args.port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #Not using
        #self.host = socket.gethostname()

        self.server_socket.bind(('', self.port))
        self.server_socket.listen(1)
        self.client_socket, self.computer_hostname = self.server_socket.accept()

        self.i2c = board.I2C()
        self.pca = PCA9685(self.i2c, address=0x40)
        self.pca.frequency = 50

        self.LEFT_WHEEL = args.left_wheel
        self.RIGHT_WHEEL = args.right_wheel

        self.LEFT_LED_RED = args.left_led_red
        self.LEFT_LED_GREEN = args.left_led_green
        self.LEFT_LED_BLUE = args.left_led_blue

        self.FRONT_LED_RED = args.front_led_red
        self.FRONT_LED_GREEN = args.front_led_green
        self.FRONT_LED_BLUE = args.front_led_blue

        self.RIGHT_LED_RED = args.right_led_red
        self.RIGHT_LED_GREEN = args.right_led_green
        self.RIGHT_LED_BLUE = args.right_led_blue

        self.TOP_LED_RED = args.top_led_red
        self.TOP_LED_GREEN = args.top_led_green
        self.TOP_LED_BLUE = args.top_led_blue

        self.all_leds_off()

        #ping command from user.  Sets the PHUC to automatically ping each quadrant.
        self.ping_active = False
        #logic command to stop pinging once one cycle completes
        #resumes after a short period
        self.ping_cycling = False

        self.ping_time = time.time()
        #ping distance in mm
        self.ping_distance = 0

        self.LUQ_TRIG = args.left_upper_quadrant_trigger
        self.LUQ_ECHO = args.left_upper_quadrant_echo
        GPIO.setup(self.LUQ_TRIG, GPIO.OUT)
        GPIO.setup(self.LUQ_ECHO, GPIO.IN)
        GPIO.output(self.LUQ_TRIG, False)
        self.pinging_luq = False
        self.luq_ping_cycle = 0
        self.ping_luq_start = 0
        self.ping_luq_stop = 0

        self.RUQ_TRIG = args.right_upper_quadrant_trigger
        self.RUQ_ECHO = args.right_upper_quadrant_echo
        GPIO.setup(self.RUQ_TRIG, GPIO.OUT)
        GPIO.setup(self.RUQ_ECHO, GPIO.IN)
        GPIO.output(self.RUQ_TRIG, False)
        self.pinging_ruq = False
        self.ruq_ping_cycle = 0
        self.ping_ruq_start = 0
        self.ping_ruq_stop = 0

        self.LLQ_TRIG = args.left_lower_quadrant_trigger
        self.LLQ_ECHO = args.left_lower_quadrant_echo
        GPIO.setup(self.LLQ_TRIG, GPIO.OUT)
        GPIO.setup(self.LLQ_ECHO, GPIO.IN)
        GPIO.output(self.LLQ_TRIG, False)
        self.pinging_llq = False
        self.llq_ping_cycle = 0
        self.ping_llq_start = 0
        self.ping_llq_stop = 0

        self.RLQ_TRIG = args.right_lower_quadrant_trigger
        self.RLQ_ECHO = args.right_lower_quadrant_echo
        GPIO.setup(self.RLQ_TRIG, GPIO.OUT)
        GPIO.setup(self.RLQ_ECHO, GPIO.IN)
        GPIO.output(self.RLQ_TRIG, False)
        self.pinging_rlq = False
        self.rlq_ping_cycle = 0
        self.ping_rlq_start = 0
        self.ping_rlq_stop = 0

        #servos FS90R 360
        #duty cycle 50hz, 700us min_pulse, 1500 neutral, 2300 max_pulse
        #self.left_wheel = servo.Servo(self.pca.channels[self.LEFT_WHEEL], min_pulse=700, max_pulse=2300)
        self.left_wheel = servo.Servo(self.pca.channels[self.LEFT_WHEEL], min_pulse=1000, max_pulse=2000)
        self.left_wheel_neutral = args.left_wheel_neutral
        self.left_wheel_speed = args.left_wheel_neutral
        self.left_wheel_target_speed = args.left_wheel_neutral
        self.left_wheel_acceleration = args.left_wheel_acceleration
        self.left_wheel.angle = self.left_wheel_speed

        self.right_wheel = servo.Servo(self.pca.channels[self.RIGHT_WHEEL], min_pulse=1000, max_pulse=2000)
        self.right_wheel_neutral = args.right_wheel_neutral
        self.right_wheel_speed = args.right_wheel_neutral
        self.right_wheel_target_speed = args.right_wheel_neutral
        self.right_wheel_acceleration = args.right_wheel_acceleration
        self.right_wheel.angle = self.right_wheel_speed

        #Camera parameters
        self.camera_on = False
        self.camera_width = args.camera_width
        self.camera_height = args.camera_height 
        # Need to implement in future      
        # resolution determines pixel skipping to form array 1-100?
        # if resolution is 1 then it sends every pixel
        # if resolution 10 then it sends every 10th pixel
        self.camera_resolution = args.camera_resolution
        # sets top corner of camera.  Can move camera location around
        self.camera_location = [args.camera_x_location, args.camera_y_location]
        self.frames_per_second = args.camera_frames_per_second
        self.camera_frame_rate = 1/self.frames_per_second

        self.picam2 = Picamera2()
        #self.picam2.preview_configuration.main.size = (1280, 720)
        self.picam2.preview_configuration.main.size = (self.camera_width, self.camera_height)
        #Sending over wifi
        self.picam2.preview_configuration.main.format = 'BGR888'
        #graycale first member of array [Y U V] just Y?
        #self.picam2.preview_configuration.main.format = 'YUV420'
        #self.camera_buffer = np.zeros((self.camera_height, self.camera_width),dtype=np.int8)
        #self.camera_buffer_grayscale = self.camera_buffer[:self.camera_height,]

        self.picam2.preview_configuration.raw.format = 'SBGGR8'
        self.camera_buffer = np.zeros((self.camera_height, self.camera_width, 3),dtype=np.uint8)

        #Flash LEDs green to indicate ready
        for i in range(5):
            self.set_pca_channel(self.LEFT_LED_GREEN, 0xFFFF)
            self.set_pca_channel(self.RIGHT_LED_GREEN, 0xFFFF)
            self.set_pca_channel(self.TOP_LED_GREEN, 0xFFFF)
            time.sleep(0.5)
            self.set_pca_channel(self.LEFT_LED_GREEN, 0x0000)
            self.set_pca_channel(self.RIGHT_LED_GREEN, 0x0000)
            self.set_pca_channel(self.TOP_LED_GREEN, 0x0000)
            time.sleep(0.5)

    def set_pca_channel(self, channel, duty_cycle):
        try:
            self.pca.channels[channel].duty_cycle = duty_cycle
        except:
            print('Error: channel: ' + str(channel) + ' duty cycle: ' + str(duty_cycle))
            self.pca = PCA9685(self.i2c, address=0x40)
            self.pca.frequency = 50
            self.reset()

    def reset(self):
        self.left_wheel_target_speed = self.left_wheel_neutral
        self.left_wheel_speed = self.left_wheel_neutral
        self.left_wheel.angle = self.left_wheel_neutral

        self.right_wheel_target_speed = self.right_wheel_neutral
        self.right_wheel_speed = self.right_wheel_neutral
        self.right_wheel.angle = self.right_wheel_neutral

        self.all_leds_off()
        self.reset_ping()
        self.stop_camera()

    def all_leds_off(self):
        self.set_pca_channel(self.LEFT_LED_RED, 0x0000)
        self.set_pca_channel(self.LEFT_LED_BLUE, 0x0000)
        self.set_pca_channel(self.LEFT_LED_GREEN, 0x0000)

        self.set_pca_channel(self.FRONT_LED_RED, 0x0000)
        self.set_pca_channel(self.FRONT_LED_BLUE, 0x0000)
        self.set_pca_channel(self.FRONT_LED_GREEN, 0x0000)

        self.set_pca_channel(self.RIGHT_LED_RED, 0x0000)
        self.set_pca_channel(self.RIGHT_LED_BLUE, 0x0000)
        self.set_pca_channel(self.RIGHT_LED_GREEN, 0x0000)

        self.set_pca_channel(self.TOP_LED_RED, 0x0000)
        self.set_pca_channel(self.TOP_LED_BLUE, 0x0000)
        self.set_pca_channel(self.TOP_LED_GREEN, 0x0000)

    def change_left_speed(self):
        if (self.left_wheel_target_speed > self.left_wheel_speed):
            self.left_wheel_speed += self.left_wheel_acceleration
            if(self.left_wheel_speed > self.left_wheel_target_speed):
                self.left_wheel_speed = self.left_wheel_target_speed
        elif (self.left_wheel_target_speed < self.left_wheel_speed):
            self.left_wheel_speed -= self.left_wheel_acceleration
            if(self.left_wheel_speed < self.left_wheel_target_speed):
                self.left_wheel_speed = self.left_wheel_target_speed
        self.left_wheel.angle = self.left_wheel_speed

    def change_right_speed(self):
        if (self.right_wheel_target_speed > self.right_wheel_speed):
            self.right_wheel_speed += self.right_wheel_acceleration
            if (self.right_wheel_speed > self.right_wheel_target_speed):
                self.right_wheel_speed = self.right_wheel_target_speed
        elif (self.right_wheel_target_speed < self.right_wheel_speed):
            self.right_wheel_speed -= self.right_wheel_acceleration
            if (self.right_wheel_speed < self.right_wheel_target_speed):
                self.right_wheel_speed = self.right_wheel_target_speed
        self.right_wheel.angle = self.right_wheel_speed

    def reset_ping(self):
        self.ping_active = False
        self.ping_cycling = False

        GPIO.output(self.LUQ_TRIG, False)
        self.pinging_luq = False
        self.luq_ping_cycle = 0
        self.ping_luq_start = 0
        self.ping_luq_stop = 0

        GPIO.output(self.RUQ_TRIG, False)
        self.pinging_ruq = False
        self.ruq_ping_cycle = 0
        self.ping_ruq_start = 0
        self.ping_ruq_stop = 0

        GPIO.output(self.LLQ_TRIG, False)
        self.pinging_llq = False
        self.llq_ping_cycle = 0
        self.ping_llq_start = 0
        self.ping_llq_stop = 0

        GPIO.output(self.RLQ_TRIG, False)
        self.pinging_rlq = False
        self.rlq_ping_cycle = 0
        self.ping_rlq_start = 0
        self.ping_rlq_stop = 0

    def ping_update(self):
        if (self.pinging_luq):
            # Time segment TRIG high ECHO low
            if (self.luq_ping_cycle == 0):
                if ((time.time() - self.ping_luq_stop) > 0.00001):
                    GPIO.output(self.LUQ_TRIG, False)
                    self.luq_ping_cycle = 1
            # Time segment TRIG low ECHO low
            elif (self.luq_ping_cycle == 1):
                self.ping_luq_start = time.time()
                if GPIO.input(self.LUQ_ECHO):
                    self.luq_ping_cycle = 2
                elif ((time.time() - self.ping_luq_stop) > 0.01):
                    self.pinging_luq = False
                    self.ping_activate_quadrant(2)
                    self.luq_ping_cycle = 0
                    sys.stdout.write('plu9999\n')
            # Time segmebnt TRIG low ECHO high
            elif (self.luq_ping_cycle == 2):
                self.ping_luq_stop = time.time()
                if (not GPIO.input(self.LUQ_ECHO)):
                    self.pinging_luq = False
                    self.ping_activate_quadrant(2)
                    self.luq_ping_cycle = 0
                    self.ping_distance = int((self.ping_luq_stop - self.ping_luq_start) * 171500)
                    sys.stdout.write('plu' + str(self.ping_distance) +'\n')
                elif ((time.time() - self.ping_luq_start) > 0.01):
                    self.pinging_luq = False
                    self.ping_activate_quadrant(2)
                    self.luq_ping_cycle = 0
                    sys.stdout.write('plu9999\n')
            else:
                self.pinging_luq = False
                self.ping_activate_quadrant(2)
                self.luq_ping_cycle = 0
                sys.stdout.write('plu9999\n')

        elif (self.pinging_ruq):
            # Time segment TRIG high ECHO low
            if (self.ruq_ping_cycle == 0):
                if ((time.time() - self.ping_ruq_stop) > 0.00001):
                    GPIO.output(self.RUQ_TRIG, False)
                    self.ruq_ping_cycle = 1
            # Time segment TRIG low ECHO low
            elif (self.ruq_ping_cycle == 1):
                self.ping_ruq_start = time.time()
                if GPIO.input(self.RUQ_ECHO):
                    self.ruq_ping_cycle = 2
                elif ((time.time() - self.ping_ruq_stop) > 0.01):
                    self.pinging_ruq = False
                    self.ping_activate_quadrant(3)
                    self.ruq_ping_cycle = 0
                    sys.stdout.write('pru9999\n')
            # Time segmebnt TRIG low ECHO high
            elif (self.ruq_ping_cycle == 2):
                self.ping_ruq_stop = time.time()
                if (not GPIO.input(self.RUQ_ECHO)):
                    self.pinging_ruq = False
                    self.ping_activate_quadrant(3)
                    self.ruq_ping_cycle = 0
                    self.ping_distance = int((self.ping_ruq_stop - self.ping_ruq_start) * 171500)
                    sys.stdout.write('pru' + str(self.ping_distance) +'\n')
                elif ((time.time() - self.ping_ruq_start) > 0.01):
                    self.pinging_ruq = False
                    self.ping_activate_quadrant(3)
                    self.ruq_ping_cycle = 0
                    sys.stdout.write('pru9999\n')
            else:
                self.pinging_ruq = False
                self.ping_activate_quadrant(3)
                self.ruq_ping_cycle = 0
                sys.stdout.write('pru9999\n')

        elif (self.pinging_llq):
            # Time segment TRIG high ECHO low
            if (self.llq_ping_cycle == 0):
                if ((time.time() - self.ping_llq_stop) > 0.00001):
                    GPIO.output(self.LLQ_TRIG, False)
                    self.llq_ping_cycle = 1
            # Time segment TRIG low ECHO low
            elif (self.llq_ping_cycle == 1):
                self.ping_llq_start = time.time()
                if GPIO.input(self.LLQ_ECHO):
                    self.llq_ping_cycle = 2
                elif ((time.time() - self.ping_llq_stop) > 0.01):
                    self.pinging_llq = False
                    self.ping_activate_quadrant(4)
                    self.llq_ping_cycle = 0
                    sys.stdout.write('pll9999\n')
            # Time segmebnt TRIG low ECHO high
            elif (self.llq_ping_cycle == 2):
                self.ping_llq_stop = time.time()
                if (not GPIO.input(self.LLQ_ECHO)):
                    self.pinging_llq = False
                    self.ping_activate_quadrant(4)
                    self.llq_ping_cycle = 0
                    self.ping_distance = int((self.ping_llq_stop - self.ping_llq_start) * 171500)
                    sys.stdout.write('pll' + str(self.ping_distance) +'\n')
                elif ((time.time() - self.ping_llq_start) > 0.01):
                    self.pinging_llq = False
                    self.ping_activate_quadrant(4)
                    self.llq_ping_cycle = 0
                    sys.stdout.write('pll9999\n')
            else:
                self.pinging_llq = False
                self.ping_activate_quadrant(4)
                self.llq_ping_cycle = 0
                sys.stdout.write('pll9999\n')

        elif (self.pinging_rlq):
            # Time segment TRIG high ECHO low
            if (self.rlq_ping_cycle == 0):
                if ((time.time() - self.ping_rlq_stop) > 0.00001):
                    GPIO.output(self.RLQ_TRIG, False)
                    self.rlq_ping_cycle = 1
            # Time segment TRIG low ECHO low
            elif (self.rlq_ping_cycle == 1):
                self.ping_rlq_start = time.time()
                if GPIO.input(self.RLQ_ECHO):
                    self.rlq_ping_cycle = 2
                elif ((time.time() - self.ping_rlq_stop) > 0.01):
                    self.pinging_rlq = False
                    self.ping_cycling = False
                    self.rlq_ping_cycle = 0
                    sys.stdout.write('prl9999\n')
            # Time segmebnt TRIG low ECHO high
            elif (self.rlq_ping_cycle == 2):
                self.ping_rlq_stop = time.time()
                if (not GPIO.input(self.RLQ_ECHO)):
                    self.pinging_rlq = False
                    self.ping_cycling = False
                    self.rlq_ping_cycle = 0
                    self.ping_distance = int((self.ping_rlq_stop - self.ping_rlq_start) * 171500)
                    sys.stdout.write('prl' + str(self.ping_distance) +'\n')
                elif ((time.time() - self.ping_rlq_start) > 0.01):
                    self.pinging_rlq = False
                    self.ping_cycling = False
                    self.rlq_ping_cycle = 0
                    sys.stdout.write('prl9999\n')
            else:
                self.pinging_rlq = False
                self.ping_cycling = False
                self.rlq_ping_cycle = 0
                sys.stdout.write('prl9999\n')

    def ping_activate_quadrant(self, quadrant):
        if quadrant == 1:
            GPIO.output(self.LUQ_TRIG, True)
            self.pinging_luq = True
            self.luq_ping_cycle = 0
            self.ping_luq_stop = time.time()
        elif quadrant == 2:
            GPIO.output(self.RUQ_TRIG, True)
            self.pinging_ruq = True
            self.ruq_ping_cycle = 0
            self.ping_ruq_stop = time.time()
        elif quadrant == 3:
            GPIO.output(self.LLQ_TRIG, True)
            self.pinging_llq = True
            self.llq_ping_cycle = 0
            self.ping_llq_stop = time.time()
        elif quadrant == 4:
            GPIO.output(self.RLQ_TRIG, True)
            self.pinging_rlq = True
            self.rlq_ping_cycle = 0
            self.ping_rlq_stop = time.time()

    def update_camera_settings(self):
        #need to update once we get camera working
        #self.picam2.stop()
        self.picam2.preview_configuration.main.size = (self.camera_width, self.camera_height)
        #self.full_camera_buffer = np.zeros((1280,720),dtype=np.int8)
        #width = self.camera_width/self.camera_resolution
        #height = self.camera_height/self.camera_resolution
        #self.camera_buffer = np.zeros((width,height),dtype=np.int8)
        #for i in range(len(width)):
        #    for j in range(len(height)):
        #        self.camera_buffer
        #self.camera_buffer_flatten = self.camera_buffer.flatten()
        #self.cap.set(3, self.camera_width)
        #self.cap.set(4, self.camera_height)

    def start_camera(self):
        self.camera_on = True
        self.update_camera_settings()
        self.picam2.start()

    def stop_camera(self):
        self.camera_on = False
        self.picam2.stop()

    def send_camera_buffer(self):
        self.camera_buffer = self.picam2.capture_array("main")

        data = pickle.dumps(self.camera_buffer)
        self.client_socket.sendall(data)

    def parse_user_input(self, user_input):
        #print("you entered: " + user_input)
        user_input_2_chars = user_input[0:2]
        user_input_3_chars = user_input[0:3]
        # wheel speed
        if user_input_3_chars == 'lws':
            self.left_wheel_target_speed = int(user_input[3:])
        elif user_input_3_chars == 'rws':
            self.right_wheel_target_speed = int(user_input[3:])
        # ultrasound ping
        elif user_input_3_chars == 'pon': #ping on
            self.ping_active = True
        elif user_input_3_chars == 'pof': #ping off
            self.reset_ping()
        # LEDs
        elif user_input_2_chars == 'lr':
            self.set_pca_channel(self.LEFT_LED_RED, int(user_input[2:]))
        elif user_input_2_chars == 'lg':
            self.set_pca_channel(self.LEFT_LED_GREEN, int(user_input[2:]))
        elif user_input_2_chars == 'lb':
            self.set_pca_channel(self.LEFT_LED_BLUE, int(user_input[2:]))
        elif user_input_2_chars == 'rr':
            self.set_pca_channel(self.RIGHT_LED_RED, int(user_input[2:]))
        elif user_input_2_chars == 'rg':
            self.set_pca_channel(self.RIGHT_LED_GREEN, int(user_input[2:]))
        elif user_input_2_chars == 'rb':
            self.set_pca_channel(self.RIGHT_LED_BLUE, int(user_input[2:]))
        elif user_input_2_chars == 'tr':
            self.set_pca_channel(self.TOP_LED_RED, int(user_input[2:]))
        elif user_input_2_chars == 'tg':
            self.set_pca_channel(self.TOP_LED_GREEN, int(user_input[2:]))
        elif user_input_2_chars == 'tb':
            self.set_pca_channel(self.TOP_LED_BLUE, int(user_input[2:]))
        elif user_input_2_chars == 'fr':
            self.set_pca_channel(self.FRONT_LED_RED, int(user_input[2:]))
        elif user_input_2_chars == 'fg':
            self.set_pca_channel(self.FRONT_LED_GREEN, int(user_input[2:]))
        elif user_input_2_chars == 'fb':
            self.set_pca_channel(self.FRONT_LED_BLUE, int(user_input[2:]))
        # global LED
        elif user_input_2_chars == 'gr':
            self.set_pca_channel(self.LEFT_LED_RED, int(user_input[2:]))
            self.set_pca_channel(self.RIGHT_LED_RED, int(user_input[2:]))
            self.set_pca_channel(self.TOP_LED_RED, int(user_input[2:]))
        elif user_input_2_chars == 'gg':
            self.set_pca_channel(self.LEFT_LED_GREEN, int(user_input[2:]))
            self.set_pca_channel(self.RIGHT_LED_GREEN, int(user_input[2:]))
            self.set_pca_channel(self.TOP_LED_GREEN, int(user_input[2:]))
        elif user_input_2_chars == 'gb':
            self.set_pca_channel(self.LEFT_LED_BLUE, int(user_input[2:]))
            self.set_pca_channel(self.RIGHT_LED_BLUE, int(user_input[2:]))
            self.set_pca_channel(self.TOP_LED_BLUE, int(user_input[2:]))
        elif user_input_2_chars == 'go':
            self.all_leds_off()
        # wheel acceleration.  likely wont change very often
        elif user_input_3_chars == 'lwa':
            self.left_wheel_acceleration = int(user_input[3:])
        elif user_input_3_chars == 'rwa':
            self.right_wheel_acceleration = int(user_input[3:])
        elif user_input_3_chars == 'lwn':
            self.left_wheel_neutral = int(user_input[3:])
            self.left_wheel_target_speed = self.left_wheel_neutral
        elif user_input_3_chars == 'rwn':
            self.right_wheel_neutral = int(user_input[3:])
            self.right_wheel_target_speed = self.right_wheel_neutral
        #camera function
        elif user_input_2_chars == 'cm':
            if(user_input[2] == '1'):
                self.start_camera()
            elif(user_input[2] == '0'):
                self.stop_camera()
            elif (user_input[2] == 'w'):
                self.camera_width = int(user_input[3:])
            elif (user_input[2] == 'h'):
                self.camera_height = int(user_input[3:])
            elif (user_input[2] == 'r'):
                self.camera_resolution = int(user_input[3:])
            elif (user_input[2] == 'l'):
                self.camera_location = int(user_input[3:])
            elif (user_input[2] == 'f'):
                self.camera_frame_rate = 1/user_input[3:]
            elif (user_input[2] == 'u'):
                self.update_camera_settings()
        elif user_input == "quit" or user_input == "q":
            self.reset()
            quit()
        elif user_input == "reset" or user_input == "r":
            self.reset()

def get_user_input():
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().strip()
    return None

def main():
    phuc = PHUC_firmware()
    time_passed = time.time()
    wheel_delta = time.time()
    reset_time = time.time()
    camera_frame_time = time.time()
    ping_time = time.time()

    while True:
        #user_input = input("enter value (type 'quit' to exit)\n")
        time_passed = time.time()
        user_input = get_user_input()
        if user_input:
            phuc.parse_user_input(user_input)
            reset_time = time.time()
        #Allow wheels to accelerate up to target speed
        if (time_passed - wheel_delta) > 0.1:
            wheel_delta = time.time()
            if(phuc.left_wheel_speed != phuc.left_wheel_target_speed):
                phuc.change_left_speed()
            if (phuc.right_wheel_speed != phuc.right_wheel_target_speed):
                phuc.change_right_speed()
        #Send camera buffer 
        if (not phuc.ping_cycling and phuc.camera_on and ((time_passed - camera_frame_time) > phuc.camera_frame_rate)):
            phuc.send_camera_buffer()   
            camera_frame_time = time.time()
        if (phuc.ping_active):
            #cycle ping through each quadrant
            if (phuc.ping_cycling):
                phuc.ping_update()
            #Start ping cycle sequenc at desired interval
            elif (time_passed - ping_time) > 1.0:
                phuc.ping_cycling = True
                ping_time = time.time()
                phuc.ping_activate_quadrant(1)
        if (time_passed - reset_time) > 60:
            phuc.reset()
            time_passed = time.time()
            reset_time = time.time()

if __name__ == '__main__':
    main()
    print("goodbye")