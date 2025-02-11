# Created by James J. Smith 11/9/2024
# Board Burner Club
#this is core file that the main_frame.py file will call.
#This is the file you will modify to create your functions
#import neural networks and drive/control your robot

#import sys
import random
import time
import pygame
import numpy as np
import PHUC_driver
import wb_PHUC_driver
import ai_backend

# Ensure webots simulation is running (dont forget to press play in webots)
SIMULATION = True
# Set to true if implementing AI/ML/Networks fully trained or otherwise
MODEL_LOADED = True
# Set to true if AI/ML/Networks are training
TRAINING = True

XBOX_360_wired_GUID = '030003f05e040000e02000010010000'
PS4_GUID = '05009b514c050000c405000000810000'

class Nameless_BBC:
    def __init__(self):
        self.hostname = "86.75.30.9"
        self.username = "bbc0"
        self.password = "bbc0"
        self.port = 2230

        #webots timestep
        self.time_step = 30

        self.left_wheel_neutral = 95
        self.right_wheel_neutral = 95
        self.left_wheel_acceleration = 10
        self.right_wheel_acceleration = 10

        # ultrasound update interval in seconds
        self.ultrasound_update = 1

        self.camera_width = 160
        self.camera_height = 120
        self.camera_height_modified = 8
        self.camera_frames_per_second = 10

        # update
        self.update_interval = 1/self.camera_frames_per_second
        self.update_time = time.time()
        self.running = False

        if SIMULATION:
            self.phuc = wb_PHUC_driver.wb_PHUC_driver(world=0,
                                                      time_step=self.time_step,
                                                      left_wheel_acceleration=self.left_wheel_acceleration,
                                                      right_wheel_acceleration=self.right_wheel_acceleration,
                                                      ultrasound_update=self.ultrasound_update,
                                                      camera_width=self.camera_width,
                                                      camera_height=self.camera_height,
                                                      camera_height_modified=self.camera_height_modified,
                                                      camera_fps=self.camera_frames_per_second)
        else:
            self.phuc = PHUC_driver.PHUC_driver(hostname=self.hostname, username=self.username,
                                                password=self.password, port=self.port,
                                                firmware_args=self.firmware_args(),
                                                left_wheel_neutral=self.left_wheel_neutral,
                                                right_wheel_neutral=self.right_wheel_neutral,
                                                left_wheel_acceleration=self.left_wheel_acceleration,
                                                right_wheel_acceleration=self.right_wheel_acceleration,
                                                camera_width=self.camera_width,
                                                camera_height=self.camera_height,
                                                camera_height_modified=self.camera_height_modified,
                                                camera_fps=self.camera_frames_per_second)

        #self.game_controller = XBoxController(phuc=self.phuc)
        self.game_controller = Playstation4Controller(phuc=self.phuc)
        self.keyboard = Keyboard(phuc=self.phuc)

        if MODEL_LOADED:
            self.model = ai_backend.PHUCBallModelMemory(self.phuc, self.keyboard, self.game_controller, TRAINING)
            # self.model = ai_backend.PHUCBallModel(self.phuc, self.keyboard, self.game_controller, TRAINING)

    # Change code to suit your build
    def firmware_args(self):
        # PCA9685 channel connections
        arg1 = ' --left_led_red 2'
        #arg2 = ' --left_led_green 1'
        arg3 = ' --left_led_blue 0'
        arg4 = ' --top_led_red 7'
        #arg5 = ' --top_led_green 6'
        arg6 = ' --top_led_blue 5'
        # arg7 = ' --front_led_red 9'
        # arg8 = ' --front_led_green 10'
        # arg9= ' --front_led_blue 11'
        # arg10 = ' --right_led_red 13'
        # arg11 = ' --right_led_green 14'
        # arg12 = ' --right_led_blue 15'
        arg13 = ' --left_wheel 4'
        arg14 = ' --right_wheel 3'

        # Raspberry PI GPIO pin locations
        # arg15 = ' --left_upper_quadrant_trigger 20'
        # arg16 = ' --left_upper_quadrant_echo 21'
        # arg17 = ' --right_upper_quadrant_trigger 23'
        # arg18 = ' --right_upper_quadrant_echo 24'
        # arg19 = ' --left_lower_quadrant_trigger 19'
        # arg20 = ' --left_lower_quadrant_echo 26'
        # arg21 = ' --right_lower_quadrant_trigger 27'
        # arg22 = ' --right_lower_quadrant_echo 22'

        arg23 = ' --left_wheel_neutral ' + str(self.left_wheel_neutral)
        arg24 = ' --right_wheel_neutral ' + str(self.right_wheel_neutral)
        # arg25 = ' --left_wheel_acceleration ' + str(self.left_wheel_acceleration)
        # arg26 = ' --right_wheel_acceleration ' + str(self.right_wheel_acceleration)

        # camera settings
        #arg27 = ' --camera_width ' + str(self.camera_width)
        #arg28 = ' --camera_height ' + str(self.camera_height)
        # arg29 = ' --camera_x_location 0'
        # arg30 = ' --camera_y_location 0'
        # arg31 = ' --camera_resolution 1'
        arg32 = ' --camera_frames_per_second ' + str(self.camera_frames_per_second)
        arg33 = ' --port ' + str(self.port)

        args = arg1 + arg3 + arg4 + arg6 + arg13 + arg14 + arg23 + arg24 + arg32 + arg33
        return args

    #Updates on global clock cycle.  Pygame operating at 30 ticks/fps will call this each time.
    #Cycle camera, update AI, neural networks, controller joystick positions, etc
    def update(self, joysticks):
        self.keyboard.update()
        if joysticks:
            collision_magnitude = 0.0
            if SIMULATION and self.phuc.world.collision:
                collision_magnitude = self.phuc.world.collision_magnitude
            self.game_controller.update(joysticks[0], collision_magnitude)
        if self.phuc.running:
            if MODEL_LOADED:
                if (time.time() - self.update_time) > self.update_interval:
                    self.update_time = time.time()
                    self.model.update()
            # update at 30 fps interval
            self.phuc.update()

    def reset(self):
        self.phuc.reset()

class Keyboard:
    def __init__(self, phuc):
        self.phuc = phuc
        self.left_wheel_speed = 0.0
        self.right_wheel_speed = 0.0

        self.forward = False
        self.reverse = False
        self.turn_left = False
        self.turn_right = False

        self.pinging_all_quadrants = False
        self.camera_on = False
        self.running = False
        self.keyboard_control_active = False

        self.update_interval = 1/self.phuc.camera_frames_per_second
        self.update_time = time.time()

    def input(self, event):
        if event.type == pygame.KEYDOWN:
            #print(event.key)
            if event.key == pygame.K_SPACE:
                self.start(True)
            elif event.key == pygame.K_LEFT:
                pass
            elif event.key == pygame.K_RIGHT:
                pass
            elif event.key == pygame.K_UP:
                pass
            elif event.key == pygame.K_DOWN:
                pass

            elif event.key == 48 or event.key == 1073741922:
                print("0 pressed")
            elif event.key == 49 or event.key == 1073741913:
                print("1 pressed")
            elif event.key == 50 or event.key == 1073741914:
                print("2 pressed")
            elif event.key == 51 or event.key == 1073741915:
                print("3 pressed")
            elif event.key == 52 or event.key == 1073741916:
                print("4 pressed")
            elif event.key == 53 or event.key == 1073741917:
                print("5 pressed")
            elif event.key == 54 or event.key == 1073741918:
                print("6 pressed")
            elif event.key == 55 or event.key == 1073741919:
                print("7 pressed")
            elif event.key == 56 or event.key == 1073741920:
                print("8 pressed")
            elif event.key == 57 or event.key == 1073741921:
                print("9 pressed")

            elif event.key == 97:
                # print("a pressed")
                self.turn_left = True
            elif event.key == 98:
                print("b pressed")
            elif event.key == 99:
                print("c pressed")
            elif event.key == 100:
                # print("d pressed")
                self.turn_right = True
            elif event.key == 101:
                print("e pressed")
            elif event.key == 102:
                print("f pressed")
            elif event.key == 103:
                print("g pressed")
            elif event.key == 104:
                print("h pressed")
            elif event.key == 105:
                print("i pressed")
            elif event.key == 106:
                print("j pressed")
            elif event.key == 107:
                # print("k pressed")
                self.keyboard_active_toggle(True)
            elif event.key == 108:
                print("l pressed")
            elif event.key == 109:
                print("m pressed")
            elif event.key == 110:
                print("n pressed")
            elif event.key == 111:
                print("o pressed")
            elif event.key == 112:
                print("p pressed")
            elif event.key == 113:
                print("q pressed")
            elif event.key == 114:
                # print("r pressed")
                self.reset(True)
            elif event.key == 115:
                # print("s pressed")
                self.reverse = True
            elif event.key == 116:
                print("t pressed")
            elif event.key == 117:
                print("u pressed")
            elif event.key == 118:
                print("v pressed")
            elif event.key == 119:
                # print("w pressed")
                self.forward = True
            elif event.key == 120:
                print("x pressed")
            elif event.key == 121:
                print("y pressed")
            elif event.key == 122:
                print("z pressed")
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                self.start(False)
            elif event.key == pygame.K_LEFT:
                pass
            elif event.key == pygame.K_RIGHT:
                pass
            elif event.key == pygame.K_UP:
                pass
            elif event.key == pygame.K_DOWN:
                pass

            elif event.key == 48 or event.key == 1073741922:
                print("0 released")
            elif event.key == 49 or event.key == 1073741913:
                print("1 released")
            elif event.key == 50 or event.key == 1073741914:
                print("2 released")
            elif event.key == 51 or event.key == 1073741915:
                print("3 released")
            elif event.key == 52 or event.key == 1073741916:
                print("4 released")
            elif event.key == 53 or event.key == 1073741917:
                print("5 released")
            elif event.key == 54 or event.key == 1073741918:
                print("6 released")
            elif event.key == 55 or event.key == 1073741919:
                print("7 released")
            elif event.key == 56 or event.key == 1073741920:
                print("8 released")
            elif event.key == 57 or event.key == 1073741921:
                print("9 released")

            elif event.key == 97:
                # print("a released")
                self.turn_left = False
            elif event.key == 98:
                print("b released")
            elif event.key == 99:
                print("c released")
            elif event.key == 100:
                # print("d released")
                self.turn_right = False
            elif event.key == 101:
                print("e released")
            elif event.key == 102:
                print("f released")
            elif event.key == 103:
                print("g released")
            elif event.key == 104:
                print("h released")
            elif event.key == 105:
                print("i released")
            elif event.key == 106:
                print("j released")
            elif event.key == 107:
                print("k released")
            elif event.key == 108:
                print("l released")
            elif event.key == 109:
                print("m released")
            elif event.key == 110:
                print("n released")
            elif event.key == 111:
                print("o released")
            elif event.key == 112:
                print("p released")
            elif event.key == 113:
                print("q released")
            elif event.key == 114:
                # print("r released")
                self.reset(False)
            elif event.key == 115:
                # print("s released")
                self.reverse = False
            elif event.key == 116:
                print("t released")
            elif event.key == 117:
                print("u released")
            elif event.key == 118:
                print("v released")
            elif event.key == 119:
                # print("w released")
                self.forward = False
            elif event.key == 120:
                print("x released")
            elif event.key == 121:
                print("y released")
            elif event.key == 122:
                print("z released")

    def update(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            pass
        if keys[pygame.K_RIGHT]:
            pass
        if keys[pygame.K_UP]:
            pass
        if keys[pygame.K_DOWN]:
            pass
        if keys[pygame.K_SPACE]:
            pass

        if (time.time() - self.update_time) > self.update_interval:
            self.update_time = time.time()
            self.update_phuc()

    def update_phuc(self):
        self.left_wheel_speed = self.phuc.left_wheel_speed
        self.right_wheel_speed = self.phuc.right_wheel_speed

        if self.forward:
            self.left_wheel_speed += 0.05
            self.right_wheel_speed += 0.05
        elif self.reverse:
            self.left_wheel_speed -= 0.05
            self.right_wheel_speed -= 0.05

        if self.left_wheel_speed > 0.85:
            self.left_wheel_speed = 0.85
        elif self.left_wheel_speed < -0.85:
            self.left_wheel_speed = -0.85
        if self.right_wheel_speed > 0.85:
            self.right_wheel_speed = 0.85
        elif self.right_wheel_speed < -0.85:
            self.right_wheel_speed = -0.85

        if self.turn_left:
            self.left_wheel_speed -= 0.07
            self.right_wheel_speed += 0.07
        elif self.turn_right:
            self.left_wheel_speed += 0.07
            self.right_wheel_speed -= 0.07

        if self.keyboard_control_active:
            self.phuc.set_wheel_speed('lw', self.left_wheel_speed)
            self.phuc.set_wheel_speed('rw', self.right_wheel_speed)

    def keyboard_active_toggle(self, pressed):
        if pressed:
            if not self.keyboard_control_active:
                print("keyboard control activated")
                self.keyboard_control_active = True
            else:
                print("keyboard control deactivated")
                self.keyboard_control_active = False
        else:
            pass

    def reset(self, pressed):
        if pressed:
            self.left_wheel_speed = 0.0
            self.right_wheel_speed = 0.0
            self.phuc.reset()
            self.pinging_all_quadrants = False
            self.camera_on = False
            self.running = False
            self.keyboard_control_active = False
            print("phuc reset")
        else:
            pass

    def start(self, pressed):
        if pressed and not self.running:
            self.running = True
            self.phuc.start()
            self.pinging_all_quadrants = True
            self.camera_on = True
            print("phuc start keyboard control")
        elif pressed and self.running:
            self.running = False
            self.phuc.paused()
            self.pinging_all_quadrants = False
            self.camera_on = False
            print("phuc pause")
        else:
            pass

class XBoxController:
    def __init__(self, phuc):
        self.phuc = phuc
        self.left_joystick_x = 0
        self.left_joystick_y = 0
        self.right_joystick_x = 0
        self.right_joystick_y = 0
        self.left_trigger = -1
        self.right_trigger = -1
        self.d_pad_x = 0
        self.d_pad_y = 0

        # self.plu_interval = 0
        # self.pru_interval = 0
        # self.pll_interval = 0
        # self.prl_interval = 0

        self.pinging_all_quadrants = False
        self.camera_on = False
        self.running = False

    # controller button/joystick pressed
    def input(self, joystick, event):
        if event.type == pygame.JOYBUTTONDOWN:
            print("Joystick button pressed.")
            if event.button == 0:
                self.A_button(True)
            elif event.button == 1:
                self.B_button(True)
            elif event.button == 2:
                self.X_button(True)
            elif event.button == 3:
                self.Y_button(True)
            elif event.button == 4:
                self.L_bumper(True)
            elif event.button == 5:
                self.R_bumper(True)
            elif event.button == 6:
                self.back_button(True)
            elif event.button == 7:
                self.start_button(True)
            elif event.button == 8:
                self.L_joystick_button(True)
            elif event.button == 9:
                self.R_joystick_button(True)

        if event.type == pygame.JOYBUTTONUP:
            # print("Joystick button released.")
            if event.button == 0:
                self.A_button(False)
                if joystick.rumble(0, 0.7, 500):
                    print(f"Rumble effect played on joystick {event.instance_id}")
                    # channel.send('green' + '\n')
            elif event.button == 1:
                self.B_button(False)
            elif event.button == 2:
                self.X_button(False)
            elif event.button == 3:
                self.Y_button(False)
            elif event.button == 4:
                self.L_bumper(False)
            elif event.button == 5:
                self.R_bumper(False)
            elif event.button == 6:
                self.back_button(False)
            elif event.button == 7:
                self.start_button(False)
            elif event.button == 8:
                self.L_joystick_button(False)
            elif event.button == 9:
                self.R_joystick_button(False)

    # controller button/joystick held
    def update(self, joystick, collision_magnitude):

        if collision_magnitude > 0.0:
            pass
            # low freq, high freq, duration
            # joystick.rumble(collision_magnitude/4, collision_magnitude, 100)
            #joystick.rumble(0, collision_magnitude, 100)
            # joystick.rumble(collision_magnitude, 0, 100)
        self.adjust_speed(joystick)

        # joysticks and triggers
        axes = joystick.get_numaxes()
        for i in range(axes):
            axis = joystick.get_axis(i)
            # L stick x-axis -1 left 1 right
            if i == 0:
                #print(axis)
                pass
            # L stick y-axis -1 forward, 1 backwards
            if i == 1:
                # print(axis)
                pass
            # L trigger -1 not pressed 1 fully pressed
            if i == 2:
                # print(axis)
                pass
            # R stick x-axis -1 left 1 right
            if i == 3:
                # print(axis)
                pass
            # R stick -1 forward, 1 backwards
            if i == 4:
                # print("R stick y axis: " + str(axis))
                pass
            # R trigger -1 not pressed 1 fully pressed
            elif i == 5:
                pass
                # print("R trigger axis: " + str(axis))

        buttons = joystick.get_numbuttons()
        for i in range(buttons):
            button = joystick.get_button(i)
            if i == 0 and button == 1:
                pass
                # channel.send('left_green\n')
                # channel.send('top_green\n')
                # channel.send('right_green\n')
                # green = True
            if i == 1 and button == 1:
                pass
                # channel.send('left_red\n')
                # channel.send('top_red\n')
                # channel.send('right_red\n')
            if i == 2 and button == 1:
                pass
                # channel.send('left_blue\n')
                # channel.send('top_blue\n')
                # channel.send('right_blue\n')
            if i == 3 and button == 1:
                pass
                # channel.send('front_red\n')
                # channel.send('front_green\n')
                # channel.send('front_blue\n')
            # text_print.tprint(screen, f"Button {i:>2} value: {button}")
            if i == 5 and button == 1:
                pass
                # channel.send('reset\n')
                # reset_puck()
                # channel.send('0' + str(int(base_position)) + '\n')
                # channel.send('1' + str(int(shoulder_position)) + '\n')
                # channel.send('2' + str(int(elbow_position)) + '\n')
                # channel.send('3' + str(int(hand_position)) + '\n')

        hats = joystick.get_numhats()
        # Hat position. All or nothing for direction, not a float like
        # get_axis(). Position is a tuple of int values (x, y).
        for i in range(hats):
            hat = joystick.get_hat(i)
            # print(hat)

        # Send hat value to xbox controller
        self.D_pad(joystick.get_hat(0)[0], joystick.get_hat(0)[1])

    def adjust_speed(self, joystick):
        # right trigger determines speed left joystick determines steering/direction
        if self.R_trigger(joystick) or self.L_joystick_xy(joystick):
            left_speed = (-self.left_joystick_y * abs(self.left_joystick_y)
                          + self.left_joystick_x * abs(self.left_joystick_x)) * (self.right_trigger*0.5 + 0.5)
            right_speed = (-self.left_joystick_y * abs(self.left_joystick_y)
                           - self.left_joystick_x * abs(self.left_joystick_x)) * (self.right_trigger*0.5 + 0.5)
            if left_speed < -0.99:
                left_speed = -0.99
            elif left_speed > 0.99:
                left_speed = 0.99
            if right_speed < -0.99:
                right_speed = -0.99
            elif right_speed > 0.99:
                right_speed = 0.99

            self.phuc.set_wheel_speed('lw', left_speed)
            self.phuc.set_wheel_speed('rw', right_speed)

    def A_button(self, pressed):
        if pressed:
            self.phuc.set_led('fg', 1)
            self.phuc.set_global_color(0,1,0)
            print("Green")
        else:
            self.phuc.set_led('fg', 0)
            self.phuc.set_global_color(0, 0, 0)
            print("Green off")

    def B_button(self, pressed):
        if pressed:
            self.phuc.set_led('fr', 1)
            self.phuc.set_global_color(1, 0, 0)
            print("Red")
        else:
            self.phuc.set_led('fr', 0)
            self.phuc.set_global_color(0, 0, 0)
            print("Red off")

    def X_button(self, pressed):
        if pressed:
            self.phuc.set_led('fb', 1)
            self.phuc.set_global_color(0, 0, 1)
            print("Blue")
        else:
            self.phuc.set_led('fb', 0)
            self.phuc.set_global_color(0, 0, 0)
            print("Blue off")

    def Y_button(self, pressed):
        if pressed:
            self.phuc.set_led('fr', random.random())
            self.phuc.set_led('fg', random.random())
            self.phuc.set_led('fb', random.random())
            self.phuc.set_led('lr', random.random())
            self.phuc.set_led('lg', random.random())
            self.phuc.set_led('lb', random.random())
            self.phuc.set_led('tr', random.random())
            self.phuc.set_led('tg', random.random())
            self.phuc.set_led('tb', random.random())
            self.phuc.set_led('rr', random.random())
            self.phuc.set_led('rg', random.random())
            self.phuc.set_led('rb', random.random())
            print("Random Colors")
        else:
            print("Random Colors off")
            self.phuc.set_led('fr', 0)
            self.phuc.set_led('fg', 0)
            self.phuc.set_led('fb', 0)
            self.phuc.set_global_color(0,0,0)

    def L_bumper(self, pressed):
        if pressed:
            if self.pinging_all_quadrants:
                self.phuc.ping_on_off(False)
                self.pinging_all_quadrants = False
            else:
                self.phuc.ping_on_off(True)
                self.pinging_all_quadrants = True
        else:
            pass

    def R_bumper(self, pressed):
        if pressed:
            if self.camera_on:
                self.phuc.turn_camera_off()
                self.camera_on = False
            else:
                self.phuc.turn_camera_on()
                self.camera_on = True
        else:
            pass

    def back_button(self, pressed):
        if pressed:
            self.left_joystick_x = 0
            self.left_joystick_y = 0
            self.right_joystick_x = 0
            self.right_joystick_y = 0
            self.phuc.reset()
            self.pinging_all_quadrants = False
            self.camera_on = False
            self.running = False
            print("phuc reset")
        else:
            pass

    def start_button(self, pressed):
        if pressed and not self.running:
            self.running = True
            self.phuc.start()
            self.pinging_all_quadrants = True
            self.camera_on = True
            print("phuc start")
        elif pressed and self.running:
            self.running = False
            self.phuc.paused()
            self.pinging_all_quadrants = False
            self.camera_on = False
        else:
            pass

    def L_joystick_button(self, pressed):
        if pressed:
            print("L joystick button pressed")
        else:
            print("L joystick button released")

    def R_joystick_button(self, pressed):
        if pressed:
            print("R joystick button pressed")
        else:
            print("R joystick button released")

    # -1 to 1 Y-vertical, X-horizontal
    def L_joystick_xy(self, joystick):
        change = False
        x_axis = joystick.get_axis(0)
        y_axis = joystick.get_axis(1)

        if abs(x_axis) <= 0.26:
            x_axis = 0.0
        if abs(self.left_joystick_x - x_axis) > 0.1:
            self.left_joystick_x = x_axis
            change = True

        if abs(y_axis) <= 0.26:
            y_axis = 0.0
        if abs(self.left_joystick_y - y_axis) > 0.1:
            self.left_joystick_y = y_axis
            change = True

        return change

    # -1 to 1 Y-vertical, X-horizontal
    def R_joystick_xy(self, joystick):
        change = False
        # Linux
        x_axis = joystick.get_axis(3)
        y_axis = joystick.get_axis(4)
        # Windows
        # x_axis = joystick.get_axis(2)
        # y_axis = joystick.get_axis(3)

        if abs(x_axis) <= 0.26:
            x_axis = 0.0
        if abs(self.right_joystick_x - x_axis) > 0.1:
            self.right_joystick_x = x_axis
            change = True

        if abs(y_axis) <= 0.26:
            y_axis = 0.0
        if abs(self.right_joystick_y - y_axis) > 0.1:
            self.right_joystick_y = y_axis
            change = True

        return change

    # -1 to 1
    def L_trigger(self, joystick):
        change = False
        # linux
        position = joystick.get_axis(2)
        # windows
        #position = self.L_trigger(joystick.get_axis(4))
        if self.left_trigger != position:
            self.left_trigger = position
            change = True
            # print("L trigger position: " + str(self.left_trigger))
        return change

    # -1 to 1
    def R_trigger(self, joystick):
        change = False
        position = joystick.get_axis(5)
        if self.right_trigger != position:
            self.right_trigger = position
            change = True
            # print("R trigger position: " + str(self.right_trigger))
        return change

    #  x-horizontal, y-vertical, values either -1 or 1
    def D_pad(self, x, y):
        # D pad left
        if x == -1:
           print("D pad left")
        # D pad right
        elif x == 1:
            print("D pad right")
        # D pad up
        if y == 1:
            print("D pad up")
        # D pad down
        if y == -1:
            print("D pad down")

class Playstation4Controller:
    def __init__(self, phuc):
        self.phuc = phuc
        self.left_joystick_x = 0
        self.left_joystick_y = 0
        self.right_joystick_x = 0
        self.right_joystick_y = 0
        self.left_trigger = -1
        self.right_trigger = -1
        self.d_pad_x = 0
        self.d_pad_y = 0

        # self.plu_interval = 0
        # self.pru_interval = 0
        # self.pll_interval = 0
        # self.prl_interval = 0

        self.pinging_all_quadrants = False
        self.camera_on = False
        self.running = False

    # controller button/joystick pressed
    def input(self, joystick, event):
        if event.type == pygame.JOYBUTTONDOWN:
            print("Joystick button pressed.")
            if event.button == 0:
                self.x_button(True)
            elif event.button == 1:
                self.circle_button(True)
            elif event.button == 2:
                self.triangle_button(True)
            elif event.button == 3:
                self.square_button(True)
            elif event.button == 4:
                self.L_bumper(True)
            elif event.button == 5:
                self.R_bumper(True)
            elif event.button == 6:
                self.L_trigger_pressed(True)
            elif event.button == 7:
                self.R_trigger_pressed(True)
            elif event.button == 8:
                self.share_button(True)
            elif event.button == 9:
                self.options_button(True)
            elif event.button == 10:
                self.ps_button(True)
            elif event.button == 11:
                self.L_joystick_button(True)
            elif event.button == 12:
                self.R_joystick_button(True)

        if event.type == pygame.JOYBUTTONUP:
            # print("Joystick button released.")
            if event.button == 0:
                self.x_button(False)
                if joystick.rumble(0, 0.7, 500):
                    print(f"Rumble effect played on joystick {event.instance_id}")
                    # channel.send('green' + '\n')
            elif event.button == 1:
                self.circle_button(False)
            elif event.button == 2:
                self.triangle_button(False)
            elif event.button == 3:
                self.square_button(False)
            elif event.button == 4:
                self.L_bumper(False)
            elif event.button == 5:
                self.R_bumper(False)
            elif event.button == 6:
                self.L_trigger_pressed(False)
            elif event.button == 7:
                self.R_trigger_pressed(False)
            elif event.button == 8:
                self.share_button(False)
            elif event.button == 9:
                self.options_button(False)
            elif event.button == 10:
                self.ps_button(False)
            elif event.button == 11:
                self.L_joystick_button(False)
            elif event.button == 12:
                self.R_joystick_button(False)

    # controller button/joystick held
    def update(self, joystick, collision_magnitude):

        if collision_magnitude > 0.0:
            pass
            # low freq, high freq, duration
            # joystick.rumble(collision_magnitude/4, collision_magnitude, 100)
            # joystick.rumble(0, collision_magnitude, 200)
            # joystick.rumble(collision_magnitude, 0, 100)
        self.adjust_speed(joystick)

        # joysticks and triggers
        axes = joystick.get_numaxes()
        for i in range(axes):
            axis = joystick.get_axis(i)
            # L stick x-axis -1 left 1 right
            if i == 0:
                # print(axis)
                pass
            # L stick y-axis -1 forward, 1 backwards
            if i == 1:
                # print(axis)
                pass
            # L trigger -1 not pressed 1 fully pressed
            if i == 2:
                # print(axis)
                pass
            # R stick x-axis -1 left 1 right
            if i == 3:
                # print(axis)
                pass
            # R stick -1 forward, 1 backwards
            if i == 4:
                # print("R stick y axis: " + str(axis))
                pass
            # R trigger -1 not pressed 1 fully pressed
            elif i == 5:
                pass
                # print("R trigger axis: " + str(axis))

        buttons = joystick.get_numbuttons()
        for i in range(buttons):
            button = joystick.get_button(i)
            if i == 0 and button == 1:
                pass
                # channel.send('left_green\n')
                # channel.send('top_green\n')
                # channel.send('right_green\n')
                # green = True
            if i == 1 and button == 1:
                pass
                # channel.send('left_red\n')
                # channel.send('top_red\n')
                # channel.send('right_red\n')
            if i == 2 and button == 1:
                pass
                # channel.send('left_blue\n')
                # channel.send('top_blue\n')
                # channel.send('right_blue\n')
            if i == 3 and button == 1:
                pass
                # channel.send('front_red\n')
                # channel.send('front_green\n')
                # channel.send('front_blue\n')
            # text_print.tprint(screen, f"Button {i:>2} value: {button}")
            if i == 5 and button == 1:
                pass
                # channel.send('reset\n')
                # reset_puck()
                # channel.send('0' + str(int(base_position)) + '\n')
                # channel.send('1' + str(int(shoulder_position)) + '\n')
                # channel.send('2' + str(int(elbow_position)) + '\n')
                # channel.send('3' + str(int(hand_position)) + '\n')

        hats = joystick.get_numhats()
        # Hat position. All or nothing for direction, not a float like
        # get_axis(). Position is a tuple of int values (x, y).
        for i in range(hats):
            hat = joystick.get_hat(i)
            # print(hat)

        # Send hat value to xbox controller
        self.D_pad(joystick.get_hat(0)[0], joystick.get_hat(0)[1])

    def adjust_speed(self, joystick):
        # right trigger determines speed left joystick determines steering/direction
        if self.R_trigger(joystick) or self.L_joystick_xy(joystick):
            left_speed = (-self.left_joystick_y * abs(self.left_joystick_y)
                          + self.left_joystick_x * abs(self.left_joystick_x)) * (self.right_trigger * 0.5 + 0.5)
            right_speed = (-self.left_joystick_y * abs(self.left_joystick_y)
                           - self.left_joystick_x * abs(self.left_joystick_x)) * (self.right_trigger * 0.5 + 0.5)
            if left_speed < -0.92:
                left_speed = -0.92
            elif left_speed > 0.92:
                left_speed = 0.92
            if right_speed < -0.92:
                right_speed = -0.92
            elif right_speed > 0.92:
                right_speed = 0.92

            self.phuc.set_wheel_speed('lw', left_speed)
            self.phuc.set_wheel_speed('rw', right_speed)

    def x_button(self, pressed):
        if pressed:
            self.phuc.set_led('fg', 1)
            self.phuc.set_global_color(0, 0, 1)
            print("Green")
        else:
            self.phuc.set_led('fg', 0)
            self.phuc.set_global_color(0, 0, 0)
            print("Green off")

    def circle_button(self, pressed):
        if pressed:
            self.phuc.set_led('fr', 1)
            self.phuc.set_global_color(1, 0, 0)
            print("Red")
        else:
            self.phuc.set_led('fr', 0)
            self.phuc.set_global_color(0, 0, 0)
            print("Red off")

    def triangle_button(self, pressed):
        if pressed:
            self.phuc.set_led('fb', 1)
            self.phuc.set_global_color(0, 1, 0)
            print("Blue")
        else:
            self.phuc.set_led('fb', 0)
            self.phuc.set_global_color(0, 0, 0)
            print("Blue off")

    def square_button(self, pressed):
        if pressed:
            self.phuc.set_led('fr', random.random())
            self.phuc.set_led('fg', random.random())
            self.phuc.set_led('fb', random.random())
            self.phuc.set_led('lr', random.random())
            self.phuc.set_led('lg', random.random())
            self.phuc.set_led('lb', random.random())
            self.phuc.set_led('tr', random.random())
            self.phuc.set_led('tg', random.random())
            self.phuc.set_led('tb', random.random())
            self.phuc.set_led('rr', random.random())
            self.phuc.set_led('rg', random.random())
            self.phuc.set_led('rb', random.random())
            print("Random Colors")
        else:
            print("Random Colors off")
            self.phuc.set_led('fr', 0)
            self.phuc.set_led('fg', 0)
            self.phuc.set_led('fb', 0)
            self.phuc.set_global_color(0, 0, 0)

    def ps_button(self, pressed):
        if pressed:
            print("ps button presse")
        else:
            pass

    def L_bumper(self, pressed):
        if pressed:
            if self.pinging_all_quadrants:
                self.phuc.ping_on_off(False)
                self.pinging_all_quadrants = False
            else:
                self.phuc.ping_on_off(True)
                self.pinging_all_quadrants = True
        else:
            pass

    def R_bumper(self, pressed):
        if pressed:
            if self.camera_on:
                self.phuc.turn_camera_off()
                self.camera_on = False
            else:
                self.phuc.turn_camera_on()
                self.camera_on = True
        else:
            pass

    def L_trigger_pressed(self, pressed):
        if pressed:
            print("L trigger pressed")
        else:
            pass

    def R_trigger_pressed(self, pressed):
        if pressed:
            print("R trigger pressed")
        else:
            pass

    # reset
    def share_button(self, pressed):
        if pressed:
            self.left_joystick_x = 0
            self.left_joystick_y = 0
            self.right_joystick_x = 0
            self.right_joystick_y = 0
            self.phuc.reset()
            self.pinging_all_quadrants = False
            self.camera_on = False
            self.running = False
            print("phuc reset")
        else:
            pass

    def options_button(self, pressed):
        if pressed and not self.running:
            self.running = True
            self.phuc.start()
            self.pinging_all_quadrants = True
            self.camera_on = True
            print("phuc start")
        elif pressed and self.running:
            self.running = False
            self.phuc.paused()
            self.pinging_all_quadrants = False
            self.camera_on = False
        else:
            pass

    def L_joystick_button(self, pressed):
        if pressed:
            print("L joystick button pressed")
        else:
            print("L joystick button released")

    def R_joystick_button(self, pressed):
        if pressed:
            print("R joystick button pressed")
        else:
            print("R joystick button released")

    # -1 to 1 Y-vertical, X-horizontal
    def L_joystick_xy(self, joystick):
        change = False
        x_axis = joystick.get_axis(0)
        y_axis = joystick.get_axis(1)

        if abs(x_axis) <= 0.08:
            x_axis = 0.0
        if abs(self.left_joystick_x - x_axis) > 0.1:
            self.left_joystick_x = x_axis
            change = True

        if abs(y_axis) <= 0.03:
            y_axis = 0.0
        if abs(self.left_joystick_y - y_axis) > 0.1:
            self.left_joystick_y = y_axis
            change = True

        return change

    # -1 to 1 Y-vertical, X-horizontal
    def R_joystick_xy(self, joystick):
        change = False
        # Linux
        x_axis = joystick.get_axis(3)
        y_axis = joystick.get_axis(4)
        # Windows
        # x_axis = joystick.get_axis(2)
        # y_axis = joystick.get_axis(3)

        if abs(x_axis) <= 0.08:
            x_axis = 0.0
        if abs(self.right_joystick_x - x_axis) > 0.1:
            self.right_joystick_x = x_axis
            change = True

        if abs(y_axis) <= 0.03:
            y_axis = 0.0
        if abs(self.right_joystick_y - y_axis) > 0.1:
            self.right_joystick_y = y_axis
            change = True

        return change

    # -1 to 1
    def L_trigger(self, joystick):
        change = False
        # linux
        position = joystick.get_axis(2)
        # windows
        # position = self.L_trigger(joystick.get_axis(4))
        if self.left_trigger != position:
            self.left_trigger = position
            change = True
            # print("L trigger position: " + str(self.left_trigger))
        return change

    # -1 to 1
    def R_trigger(self, joystick):
        change = False
        position = joystick.get_axis(5)
        if self.right_trigger != position:
            self.right_trigger = position
            change = True
            # print("R trigger position: " + str(self.right_trigger))
        return change

    #  x-horizontal, y-vertical, values either -1 or 1
    def D_pad(self, x, y):
        # D pad left
        if x == -1:
            print("D pad left")
        # D pad right
        elif x == 1:
            print("D pad right")
        # D pad up
        if y == 1:
            print("D pad up")
        # D pad down
        if y == -1:
            print("D pad down")

#Keeps SSH open
def keep_alive(channel):
    channel.send("\n")  # Send a newline to keep the channel active
    # time.sleep(60)  # Adjust the sleep time as needed

def reset():
    global left_wheel
    global right_wheel
    global left_wheel_direction
    global right_wheel_direction
    global red
    global green
    global blue

    left_wheel = 95
    right_wheel = 95
    left_wheel_direction = 0
    right_wheel_direction = 0
    red = False
    green = False
    blue = False

# This is a simple class that will help us print to the screen.
# It has nothing to do with the joysticks, just outputting the
# information.
class TextPrint:
    def __init__(self):
        self.reset()
        self.font = pygame.font.Font(None, 25)

    def tprint(self, screen, text):
        text_bitmap = self.font.render(text, True, (0, 0, 0))
        screen.blit(text_bitmap, (self.x, self.y))
        self.y += self.line_height

    def reset(self):
        self.x = 10
        self.y = 10
        self.line_height = 15

    def indent(self):
        self.x += 10

    def unindent(self):
        self.x -= 10

def main():
    pygame.init()

    global left_wheel
    global right_wheel
    global left_wheel_ll
    global left_wheel_ul
    global right_wheel_ll
    global right_wheel_ul

    nameless_bbc = Nameless_BBC()
    #phuc = PHUC_driver.PHUC_driver(left_wheel_neutral=LEFT_WHEEL_NEUTRAL,
    #                               right_wheel_neutral=RIGHT_WHEEL_NEUTRAL,
    #                               left_wheel_acceleration=LEFT_WHEEL_ACCELERATION,
    #                               right_wheel_acceleration=RIGHT_WHEEL_ACCELERATION)
    #xbox_controller = XBoxController(phuc=phuc)
    left_wheel = nameless_bbc.left_wheel_neutral
    right_wheel = nameless_bbc.right_wheel_neutral
    left_wheel_direction = nameless_bbc.left_wheel_neutral
    right_wheel_direction = nameless_bbc.right_wheel_neutral
    green = 0
    red = 0
    blue = 0

    # Set the width and height of the screen (width, height), and name the window.
    screen = pygame.display.set_mode((1100,1280))
    # screen = pygame.display.set_mode((1100, 500))
    pygame.display.set_caption("Nameless_BBC")

    # Used to manage how fast the screen updates.
    clock = pygame.time.Clock()

    # Get ready to print.
    text_print = TextPrint()

    # This dict can be left as-is, since pygame will generate a
    # pygame.JOYDEVICEADDED event for every joystick connected
    # at the start of the program.
    joysticks = {}

    running = True
    while running:
        # Event processing step.
        # Possible joystick events: JOYAXISMOTION, JOYBALLMOTION, JOYBUTTONDOWN,
        # JOYBUTTONUP, JOYHATMOTION, JOYDEVICEADDED, JOYDEVICEREMOVED
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False  # Flag that we are done so we exit this loop.

            if event.type == pygame.KEYDOWN or event.type == pygame.KEYUP:
                nameless_bbc.keyboard.input(event)

            if event.type == pygame.JOYBUTTONUP or event.type == pygame.JOYBUTTONDOWN:
                if event.instance_id == 0:
                    nameless_bbc.game_controller.input(joysticks[0], event)

            # Handle hotplugging
            if event.type == pygame.JOYDEVICEADDED:
                # This event will be generated when the program starts for every
                # joystick, filling up the list without needing to create them manually.
                joy = pygame.joystick.Joystick(event.device_index)
                joysticks[joy.get_instance_id()] = joy
                # print(f"Joystick {joy.get_instance_id()} connencted")

            if event.type == pygame.JOYDEVICEREMOVED:
                del joysticks[event.instance_id]
                # print(f"Joystick {event.instance_id} disconnected")

        # nameless_bbc.phuc.update()
        # Updates program including the controller update function
        # nameless_bbc.update(joysticks[0])
        nameless_bbc.update(joysticks)

        # Drawing step
        # First, clear the screen to white. Don't put other drawing commands
        # above this, or they will be erased with this command.
        screen.fill((255, 255, 255))

        # phuc camera
        my_surface = pygame.pixelcopy.make_surface(nameless_bbc.phuc.camera_buffer)
        surf = pygame.transform.scale(my_surface, (nameless_bbc.camera_width * 4, nameless_bbc.camera_height * 4))
        screen.blit(surf, (410, 20))

        #array = np.vstack((nameless_bbc.phuc.world.perception_field,
        #                   nameless_bbc.phuc.world.perception_field))

        #wb_perception_surface = pygame.pixelcopy.make_surface(array)
        wb_perception_surface = pygame.pixelcopy.make_surface(np.transpose(nameless_bbc.phuc.world.perception_field_surface, (1,0,2)))
        wb_perc_surf = pygame.transform.scale(wb_perception_surface, (nameless_bbc.camera_width*4, 10))
        screen.blit(wb_perc_surf, (410, 10))

        # Black background for nn representation
        pygame.draw.rect(screen, (0,0,0), pygame.Rect(54, 505,(nameless_bbc.camera_height+1)*8+1,(nameless_bbc.camera_height+1)*6+16))
        # square segments separated out to show individual frames sent to nn
        # high density central segments not represented
        for i in range(8):
            for j in range(6):
                sub_surf = pygame.pixelcopy.make_surface(nameless_bbc.phuc.camera_buffer[20*i : 20+20*i,20*j : 20+20*j,:])
                _surf = pygame.transform.scale(sub_surf, (nameless_bbc.camera_height, nameless_bbc.camera_height))
                # screen.blit(_surf, (410, 500))
                screen.blit(_surf, (55 + (nameless_bbc.camera_height+1)*i, 521 + (nameless_bbc.camera_height+1)*j))

        if MODEL_LOADED:
            for i in range(len(nameless_bbc.model.prediction)):
                pass

        # state perception of camera
        wb_perc_surf_nn = pygame.transform.scale(wb_perception_surface, ((nameless_bbc.camera_height+1)*8, 14))
        screen.blit(wb_perc_surf_nn, (55, 506))

        text_print.reset()

        # Get count of joysticks.
        joystick_count = pygame.joystick.get_count()

        text_print.tprint(screen, f"Number of joysticks: {joystick_count}")
        text_print.indent()

        # For each joystick:
        for joystick in joysticks.values():
            jid = joystick.get_instance_id()

            text_print.tprint(screen, f"Joystick {jid}")
            text_print.indent()

            # Get the name from the OS for the controller/joystick.
            name = joystick.get_name()
            text_print.tprint(screen, f"Joystick name: {name[0:4]}")

            guid = joystick.get_guid()
            text_print.tprint(screen, f"GUID: {guid}")

            power_level = joystick.get_power_level()
            text_print.tprint(screen, f"Joystick's power level: {power_level}")

            # Usually axis run in pairs, up/down for one, and left/right for
            # the other. Triggers count as axes.
            axes = joystick.get_numaxes()
            text_print.tprint(screen, f"Number of axes: {axes}")
            text_print.indent()

            for i in range(axes):
                axis = joystick.get_axis(i)
                text_print.tprint(screen, f"Axis {i} value: {axis:>6.3f}")
            text_print.unindent()

            buttons = joystick.get_numbuttons()
            text_print.tprint(screen, f"Number of buttons: {buttons}")
            text_print.indent()

            for i in range(buttons):
                button = joystick.get_button(i)
                text_print.tprint(screen, f"Button {i:>2} value: {button}")
            text_print.unindent()

            hats = joystick.get_numhats()
            text_print.tprint(screen, f"Number of hats: {hats}")
            text_print.indent()

            # Hat position. All or nothing for direction, not a float like
            # get_axis(). Position is a tuple of int values (x, y).
            for i in range(hats):
                hat = joystick.get_hat(i)
                text_print.tprint(screen, f"Hat {i} value: {str(hat)}")
            text_print.unindent()

            text_print.unindent()

        if MODEL_LOADED and TRAINING:
            text_print.tprint(screen, f"number of games: {nameless_bbc.model.n_games}")
            text_print.tprint(screen, f"record score: {nameless_bbc.phuc.world.record}")
            text_print.tprint(screen, f"current score: {nameless_bbc.phuc.world.points}")

        # Go ahead and update the screen with what we've drawn.
        pygame.display.flip()

        # Limit to 30 frames per second.
        clock.tick(nameless_bbc.time_step)

    print("close phuc")
    nameless_bbc.phuc.reset()
    nameless_bbc.phuc.close()

if __name__ == '__main__':
    main()
    pygame.quit()
    print("goodbye")