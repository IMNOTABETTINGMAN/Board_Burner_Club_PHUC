#PHUC_driver file created by James J. Smith 11/8/2024
import sys
import paramiko
import time
import socket
import pickle
import numpy as np
import wb_worlds
import cv2
#servos FS90R 360
# #duty cycle 50hz, 700us min_pulse, 1500 neutral, 2300 max_pulse

class PHUC_driver:
    def __init__(self, hostname, username, password, port, firmware_args,
                 left_wheel_neutral, right_wheel_neutral,
                 left_wheel_acceleration, right_wheel_acceleration,
                 camera_width, camera_height, camera_height_modified, camera_fps):

        self.world = wb_worlds.PHUC_real_world()

        self.hostname = hostname
        self.username = username
        self.password = password
        self.port = port

        self.firmware_args = firmware_args

        # Create an SSH client
        self.ssh = paramiko.SSHClient()

        # Automatically add the host key
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Connect to the Raspberry Pi
        self.ssh.connect(hostname=self.hostname, username=self.username, password=self.password)

        #self.ssh.get_transport().window_size = 3*1024*1024

        self.channel = self.ssh.invoke_shell()
        # Execute the command
        self.channel.send('python3 ./Documents/ROBOT/PHUC_firmware.py' + self.firmware_args +'\n')

        #use different socket to transmit image array
        for attempt in range(10):
            try:
                time.sleep(5)
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_socket.connect((self.hostname, self.port))
                self.client_socket.setblocking(False)
                break
            except (socket.error, ConnectionRefusedError) as e:
                print(f"Connection attempt {attempt + 1} failed: {e}")

        time.sleep(10)

        # turn all LEDs off
        self.channel.send('go\n')

        self.left_wheel_neutral = left_wheel_neutral
        self.left_wheel_acceleration = left_wheel_acceleration
        self.right_wheel_neutral = right_wheel_neutral
        self.right_wheel_acceleration = right_wheel_acceleration

        self.set_wheel_neutral('lw', self.left_wheel_neutral)
        self.set_wheel_neutral('rw', self.right_wheel_neutral)
        self.set_wheel_acceleration('lw', self.left_wheel_acceleration)
        self.set_wheel_acceleration('rw', self.right_wheel_acceleration)

        # 12 bit color 0-65535
        self.left_red = 0x0000
        self.left_green = 0x0000
        self.left_blue = 0x0000

        self.front_red = 0x0000
        self.front_green = 0x0000
        self.front_blue = 0x0000

        self.right_red = 0x0000
        self.right_green = 0x0000
        self.right_blue = 0x0000

        self.top_red = 0x0000
        self.top_green = 0x0000
        self.top_blue = 0x0000

        # actively pinging
        self.pinging_all_quadrants = False

        # distance in mm
        self.LUQ_ultrasound = 0
        self.RUQ_ultrasound = 0
        self.LLQ_ultrasound = 0
        self.RLQ_ultrasound = 0

        self.camera_on = False
        #stores returned values from camera in 1d array
        #self.camera_buffer = []
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_height_modified = camera_height_modified
        #resolution determines pixel skipping to form array 1-100?
        #if resolution is 1 then it sends every pixel
        #if resolution 10 then it sends every 10th pixel
        self.camera_resolution = 1
        #sets top corner of camera.  Can move camera location around
        self.camera_location = [0,0]
        #trying for 30.  We'll see
        self.camera_frames_per_second = camera_fps

        #self.camera_buffer_flat = np.zeros((self.camera_width*self.camera_height*3), dtype=np.uint8)
        self.camera_buffer = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
        # will need to debug
        self.camera_buffer_modified = np.zeros((self.camera_width, self.camera_height_modified, 3), dtype=np.uint8)
        #self.camera_buffer_reshape = np.reshape(self.camera_buffer,(self.camera_width, self.camera_height, 3))
        #print(self.camera_buffer.shape)
        #self.camera_buffer = np.zeros((self.camera_width, self.camera_height, 3), dtype=np.uint8)

        self.running = False

    #direction: left_upper = 'plu', right_upper = 'pru', left_lower = 'pll', right_lower = 'prl'
    def ping_direction(self, direction):
        if (direction == 'plu' and self.pinging_LUQ == False):
            pass
            #self.pinging_LUQ = True
            #self.channel.send(direction +'\n')
        elif (direction == 'pru' and self.pinging_RUQ == False):
            pass
            #self.pinging_RUQ = True
            #self.channel.send(direction +'\n')
        elif (direction == 'pll' and self.pinging_LLQ == False):
            pass
            #self.pinging_LLQ = True
            #self.channel.send(direction +'\n')
        elif (direction == 'prl' and self.pinging_RLQ == False):
            pass
            #self.pinging_RLQ = True
            #self.channel.send(direction +'\n')

    # Turns ping function on. PHUC pings in all directions at one second intervals
    def ping_on_off(self, on):
        if on:
            self.channel.send('pon\n')
        else:
            self.channel.send('pof\n')

    def ping_return(self, data):
        # print("data in ping return: " + data)
        if (data[0:3] == 'plu'):
            self.pinging_LUQ = False
            self.LUQ_ultrasound = int(data[3:])
            print("Ping Left Front: " + str(self.LUQ_ultrasound) + "mm")
        elif (data[0:3] == 'pru'):
            self.pinging_RUQ = False
            self.RUQ_ultrasound = int(data[3:])
            print("Ping Right Front: " + str(self.RUQ_ultrasound) + "mm")
        elif (data[0:3] == 'pll'):
            self.pinging_LLQ = False
            self.LLQ_ultrasound = int(data[3:])
            print("Ping Left Rear: " + str(self.LLQ_ultrasound) + "mm")
        elif (data[0:3] == 'prl'):
            self.pinging_RLQ = False
            self.RLQ_ultrasound = int(data[3:])
            print("Ping Right Rear: " + str(self.RLQ_ultrasound) + "mm")

    def camera_input(self, buffer):
        self.camera_buffer = buffer

    #led = lef_red = 'lr', top_green = 'tg', front_blue = 'fb', right_red = 'rr' etc..
    #value is 12 bit number 0 - 4095
    def set_led(self, led, value):
        value *= 65535
        if value < 0:
            value = 0
        elif value > 65535:
            value = 65535
        self.channel.send(led + str(int(value)) +'\n')

    #global color sent as 'gc' and sets left, right, and top leds to the same color
    #excludes front
    #values are float 0-1 which will map to 16 bits 0 - 65535
    def set_global_color(self, red, green, blue):
        red *= 65535
        green *= 65535
        blue *= 65535
        if red < 0:
            red = 0
        elif red > 65535:
            red = 65535
        if green < 0:
            green = 0
        elif green > 65535:
            green = 65535
        if blue < 0:
            blue = 0
        elif blue > 65535:
            blue = 65535
        self.left_red = int(red)
        self.left_green = int(green)
        self.left_blue = int(blue)

        self.right_red = int(red)
        self.right_green = int(green)
        self.right_blue = int(blue)

        self.top_red = int(red)
        self.top_green = int(green)
        self.top_blue = int(blue)

        self.channel.send('gr' + str(int(red)) +'\n')
        self.channel.send('gg' + str(int(green)) +'\n')
        self.channel.send('gb' + str(int(blue)) +'\n')

    # left wheel = 'lw' right wheel = 'rw'
    # speed from -1 to 1, changed to
    # 0-180 where 90 is stopped
    def set_wheel_speed(self, wheel, speed):
        position = 90
        if wheel == "lw":
            position = int(self.phuc.right_wheel_neutral + 90 * speed)
        elif wheel == "rw":
            position = int(self.phuc.right_wheel_neutral - 90 * speed)
        if position < 0:
            position = 0
        elif position > 180:
            position = 180
        self.channel.send(wheel + 's' + str(int(position)) +'\n')

    # left wheel = 'lw' right wheel = 'rw'
    # acceleration 1-100
    def set_wheel_acceleration(self, wheel, acceleration):
        if acceleration < 1:
            acceleration = 1
        elif acceleration > 100:
            acceleration = 100
        self.channel.send(wheel + 'a' + str(int(acceleration)) +'\n')

    # left wheel = 'lw' right wheel = 'rw'
    # neutral = 90 +/- offset
    def set_wheel_neutral(self, wheel, speed):
        if speed < 10:
            speed = 10
        elif speed > 170:
            speed = 170
        self.channel.send(wheel + 'n' + str(int(speed)) +'\n')

    #camera = 'cm' width = 'w', height = 'h', resolution = 'r', location = 'l', frame rate = 'f'
    def set_camera_parameters(self):
        self.channel.send('cmw' + str(self.camera_width) +'\n')
        self.channel.send('cmh' + str(self.camera_height) +'\n')
        self.channel.send('cmr' + str(self.camera_resolution) +'\n')
        self.channel.send('cml' + str(self.camera_location) +'\n')
        self.channel.send('cmf' + str(self.camera_frame_rate) +'\n')

    def turn_camera_on(self):
        self.channel.send('cm1\n')
        self.camera_on = True

    def turn_camera_off(self):
        self.channel.send('cm0\n')
        self.camera_on = False

    def update(self):
        #check ssh.channel
        while True:
            if self.channel.recv_ready():
                output = self.channel.recv(3*1024*1024).decode('utf-8')
                #print(output)
                #print(len(output))
                tokens = output.split('\n')
                for token in tokens:
                    #print(token)
                    if (len(token) > 5 and token[0] == 'p'):
                        #print("send this one?: " + token)
                        self.ping_return(token)
            else:
                break
        #check client_socket separate from self.channel.recv_ready()
        if self.camera_on:
            data = []
            try:
                while True:
                    packet = self.client_socket.recv(8192)
                    if not packet:
                        break
                    data.append(packet)
            except socket.error as e:
                pass

            if data:
                try:
                    array = np.array(pickle.loads(b"".join(data)))
                    self.camera_buffer = np.transpose(array, (1,0,2))
                    for i in range(self.camera_width):
                        for j in range(0, self.camera_height_modified):
                            self.camera_buffer_modified[i][j] = self.camera_buffer[i][j + self.camera_height/2 - self.camera_height_modified/2]
                except pickle.UnpicklingError as e:
                    pass
                    #print(e)
                except Exception as e:
                    print(e)

    # Information that can only be obtained from webots:
    # Collisions, absolute position of objects
    def get_state(self):
        state = []
        return np.array(state, dtype=float)

    def reset(self):
        self.left_wheel_speed = self.left_wheel_neutral
        self.right_wheel_speed = self.right_wheel_neutral

        self.set_wheel_speed('lw', self.left_wheel_neutral)
        self.set_wheel_speed('rw', self.right_wheel_neutral)
        self.set_wheel_acceleration('lw', self.left_wheel_acceleration)
        self.set_wheel_acceleration('rw', self.right_wheel_acceleration)

        self.turn_camera_off()
        self.camera_on = False

        self.set_global_color(0, 0, 0)

        self.front_red = 0
        self.front_green = 0
        self.front_blue = 0
        self.set_led('fr', self.front_red)
        self.set_led('fg', self.front_green)
        self.set_led('fb', self.front_blue)

        self.LU_ultrasound = 0
        self.RU_ultrasound = 0
        self.LL_ultrasound = 0
        self.RL_ultrasound = 0

        self.camera_buffer = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
        self.camera_buffer_modified = np.zeros((self.camera_width, self.camera_height_modified, 3), dtype=np.uint8)
        self.camera_width = 160
        self.camera_height = 120
        self.camera_resolution = 1
        self.camera_location = [0, 0]
        self.camera_frames_per_second = 10
        self.set_camera_parameters()

    # This will start whatever activity the phuc is engated in
    def start(self):
        self.turn_camera_on()
        self.camera_on = True
        self.pinging_all_quadrants = True
        self.ping_on_off(True)
        self.running = True

    def close(self):
        self.running = False
        self.channel.send('quit\n')
        self.channel.close()
        self.ssh.close()
        self.client_socket.close()