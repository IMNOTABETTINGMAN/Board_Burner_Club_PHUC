# Created by James J. Smith 11/9/2024
# Board Burner Club
#this is core file that the main_frame.py file will call.
#This is the file you will modify to create your functions
#import neural networks and drive/control your robot
#import sys
import random
import time
import pygame
import PHUC_driver

#Change class name to <your name>_BBC<your number>  eg  class Bob_BBC55:
class YourName_BBC:
    def __init__(self):
        self.hostname = "86.75.30.9" #IP address of PHUC
        self.username = "username"
        self.password = "password"
        self.port = 1111 #designate port you want to communicate through on the PHUC robot

        # Change these values as needed to ensure PHUC is stationary when initialized
        self.left_wheel_neutral = 90
        self.right_wheel_neutral = 90
        self.left_wheel_acceleration = 10
        self.right_wheel_acceleration = 10

        # default camera values
        self.camera_width = 160
        self.camera_height = 120
        self.camera_frames_per_second = 10

        self.phuc = PHUC_driver.PHUC_driver(hostname=self.hostname, username=self.username,
                                            password=self.password, port=self.port,
                                            firmware_args=self.firmware_args(),
                                            left_wheel_neutral=self.left_wheel_neutral,
                                            right_wheel_neutral=self.right_wheel_neutral,
                                            left_wheel_acceleration=self.left_wheel_acceleration,
                                            right_wheel_acceleration=self.right_wheel_acceleration,
                                            camera_width=self.camera_width,
                                            camera_height=self.camera_height,
                                            camera_fps=self.camera_frames_per_second)
        self.controller = XBox360Controller(phuc=self.phuc)

    # Change code to suit your build
    # PCA9685 channels 0-15 left-to right
    # See raspi zero image for GPIO pin numbers
    def firmware_args(self):
        # PCA9685 channel connections
        # arg1 = ' --left_led_red 0'
        # arg2 = ' --left_led_green 1'
        # arg3 = ' --left_led_blue 2'
        # arg4 = ' --top_led_red 5'
        # arg5 = ' --top_led_green 6'
        # arg6 = ' --top_led_blue 7'
        # arg7 = ' --front_led_red 9'
        # arg8 = ' --front_led_green 10'
        # arg9= ' --front_led_blue 11'
        # arg10 = ' --right_led_red 13'
        # arg11 = ' --right_led_green 14'
        # arg12 = ' --right_led_blue 15'
        # arg13 = ' --left_wheel 3'
        # arg14 = ' --right_wheel 4'

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
        arg25 = ' --left_wheel_acceleration ' + str(self.left_wheel_acceleration)
        arg26 = ' --right_wheel_acceleration ' + str(self.right_wheel_acceleration)

        # camera settings
        #arg27 = ' --camera_width ' + str(self.camera_width)
        #arg28 = ' --camera_height ' + str(self.camera_height)
        # arg29 = ' --camera_x_location 0'
        # arg30 = ' --camera_y_location 0'
        # arg31 = ' --camera_resolution 1'
        arg32 = ' --camera_frames_per_second ' + str(self.camera_frames_per_second)
        arg33 = ' --port ' + str(self.port)

        # add argument you wish to send to PHUC robot on initialization to string of arguments
        args = arg23 + arg24 + arg32 + arg33
        return args

    #Updates on global clock cycle.  Pygame operating at 30 ticks/fps will call this each time.
    #Cycle camera, update AI, neural networks, controller joystick positions, etc
    def update(self, joystick):
        self.controller.update(joystick)

    def reset(self):
        self.phuc.reset()

class XBox360Controller:
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

        self.plu_interval = 0
        self.pru_interval = 0
        self.pll_interval = 0
        self.prl_interval = 0

        self.camera_on = False

    def input(self, joystick, event):
        if event.type == pygame.JOYBUTTONDOWN:
            print("Joystick button pressed.")
            if event.button == 0:
                self.A_button(True)
                if joystick.rumble(0, 0.7, 500):
                    print(f"Rumble effect played on joystick {event.instance_id}")
                    # channel.send('green' + '\n')
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
            print("Joystick button released.")
            if event.button == 0:
                self.A_button(False)
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

    def update(self, joystick):
        # send axis information to xbox controller
        self.L_joystick_xy(joystick.get_axis(0), joystick.get_axis(1))
        # Linux
        self.R_joystick_xy(joystick.get_axis(3), joystick.get_axis(4))
        self.L_trigger(joystick.get_axis(2))
        # Windows
        # self.R_joystick_xy(joystick.get_axis(2), joystick.get_axis(3))
        # self.L_trigger(joystick.get_axis(4))

        self.R_trigger(joystick.get_axis(5))

        axes = joystick.get_numaxes()
        for i in range(axes):
            axis = joystick.get_axis(i)
            if (i == 1 and (axis > 0.30)):
                pass
                # channel.send('lws450\n')
                # channel.send('left forward\n')
                # left_wheel_direction = 1
            elif (i == 1 and (axis < -0.30)):
                pass
                # channel.send('lws150\n')
                # channel.send('left reverse\n')
                #left_wheel_direction = -1
            elif (i == 1 and (abs(axis) <= 0.30)):
                pass
                # channel.send('lws307\n')
                # channel.send('lwn\n')
                #left_wheel_direction = 0

            if (i == 3 and (axis > 0.30)):
                pass
                # channel.send('right forward\n')
                #right_wheel_direction = 1
            elif (i == 3 and (axis < -0.30)):
                pass
                # channel.send('right reverse\n')
                #right_wheel_direction = -1
            elif (i == 3 and (abs(axis) <= 0.30)):
                pass
                # channel.send('right stop\n')
                #right_wheel_direction = 0

        buttons = joystick.get_numbuttons()
        for i in range(buttons):
            button = joystick.get_button(i)
            if (i == 0 and button == 1):
                pass
                # channel.send('left_green\n')
                # channel.send('top_green\n')
                # channel.send('right_green\n')
                # green = True
            if (i == 1 and button == 1):
                pass
                # channel.send('left_red\n')
                # channel.send('top_red\n')
                # channel.send('right_red\n')
            if (i == 2 and button == 1):
                pass
                # channel.send('left_blue\n')
                # channel.send('top_blue\n')
                # channel.send('right_blue\n')
            if (i == 3 and button == 1):
                pass
                # channel.send('front_red\n')
                # channel.send('front_green\n')
                # channel.send('front_blue\n')
            # text_print.tprint(screen, f"Button {i:>2} value: {button}")
            if (i == 5 and button == 1):
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

        # Send hat value to xbox controller
        self.D_pad(joystick.get_hat(0)[0], joystick.get_hat(0)[1])

    def A_button(self, pressed):
        if pressed:
            self.phuc.set_led('fg', 65500)
            self.phuc.set_global_color(0,65500,0)
            print("Green")
        else:
            self.phuc.set_led('fg', 0)
            self.phuc.set_global_color(0, 0, 0)
            print("Green off")

    def B_button(self, pressed):
        if pressed:
            self.phuc.set_led('fr', 65500)
            self.phuc.set_global_color(65500, 0, 0)
            print("Red")
        else:
            self.phuc.set_led('fr', 0)
            self.phuc.set_global_color(0, 0, 0)
            print("Red off")

    def X_button(self, pressed):
        if pressed:
            self.phuc.set_led('fb', 65500)
            self.phuc.set_global_color(0, 0, 65500)
            print("Blue")
        else:
            self.phuc.set_led('fb', 0)
            self.phuc.set_global_color(0, 0, 0)
            print("Blue off")

    def Y_button(self, pressed):
        if pressed:
            self.phuc.set_led('fr', random.randint(0, 65500))
            self.phuc.set_led('fg', random.randint(0, 65500))
            self.phuc.set_led('fb', random.randint(0, 65500))
            self.phuc.set_led('lr', random.randint(0,65500))
            self.phuc.set_led('lg', random.randint(0, 65500))
            self.phuc.set_led('lb', random.randint(0, 65500))
            self.phuc.set_led('tr', random.randint(0, 65500))
            self.phuc.set_led('tg', random.randint(0, 65500))
            self.phuc.set_led('tb', random.randint(0, 65500))
            self.phuc.set_led('rr', random.randint(0, 65500))
            self.phuc.set_led('rg', random.randint(0, 65500))
            self.phuc.set_led('rb', random.randint(0, 65500))
            print("Random Colors")
        else:
            print("Random Colors off")
            self.phuc.set_led('fr', 0)
            self.phuc.set_led('fg', 0)
            self.phuc.set_led('fb', 0)
            self.phuc.set_global_color(0,0,0)

    def L_bumper(self, pressed):
        if pressed:
            self.phuc.ping_on_off(True)
        else:
            self.phuc.ping_on_off(False)
            print("L button released")

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
            # self.phuc.turn_camera_off()

    def back_button(self, pressed):
        if pressed:
            print("back button pressed")
        else:
            print("back button released")

    def start_button(self, pressed):
        if pressed:
            print("start button pressed")
        else:
            print("start button released")

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
    def L_joystick_xy(self, x_axis, y_axis):
        if (abs(self.left_joystick_x - x_axis) > 0.1):
            self.left_joystick_x = x_axis
            #print("L X axis: " + str(self.left_joystick_x))
        if (abs(self.left_joystick_y - y_axis) > 0.1):
            self.left_joystick_y = y_axis
            if (abs(self.left_joystick_y) > 0.26):
                position = int(self.phuc.left_wheel_neutral - 90 * self.left_joystick_y)
                if (position < 30):
                    position = 30
                elif (position > 150):
                    position = 150
                self.phuc.set_wheel_speed('lw', position)
                print("L wheel position: " + str(position))
            else:
                self.phuc.set_wheel_speed('lw', self.phuc.left_wheel_neutral)
                print("L wheel position: " + str(self.phuc.left_wheel_neutral))

    # -1 to 1 Y-vertical, X-horizontal
    def R_joystick_xy(self, x_axis, y_axis):
        if (abs(self.right_joystick_x - x_axis) > 0.1):
            self.right_joystick_x = x_axis
            #print("R X axis: " + str(self.right_joystick_x))
        if (abs(self.right_joystick_y - y_axis) > 0.1):
            self.right_joystick_y = y_axis
            if (abs(self.right_joystick_y) > 0.26):
                position = int(self.phuc.right_wheel_neutral + 90*self.right_joystick_y)
                if(position < 30):
                    position = 30
                elif (position > 150):
                    position = 150
                self.phuc.set_wheel_speed('rw', position)
                print("R wheel position: " + str(position))
            else:
                self.phuc.set_wheel_speed('rw', self.phuc.right_wheel_neutral)
                print("R wheel position: " + str(self.phuc.right_wheel_neutral))

    # -1 to 1
    def L_trigger(self, position):
        if (self.left_trigger != position):
            self.left_trigger = position
            print("L trigger position: " + str(self.left_trigger))

    # -1 to 1
    def R_trigger(self, position):
        if (self.right_trigger != position):
            self.right_trigger = position
            print("R trigger position: " + str(self.right_trigger))

    #  x-horizontal, y-vertical, values either -1 or 1
    def D_pad(self, x, y):
        if (x == -1 and y == 1):
            if((time.time() - self.plu_interval) > 1):
                self.phuc.pinging_LUQ = False
                self.plu_interval = time.time()
                self.phuc.ping_direction('plu')
        elif (x == 1 and y == 1):
            if ((time.time() - self.pru_interval) > 1):
                self.phuc.pinging_RUQ = False
                self.pru_interval = time.time()
                self.phuc.ping_direction('pru')
        elif (x == -1 and y == -1):
            if ((time.time() - self.pll_interval) > 1):
                self.phuc.pinging_LLQ = False
                self.pll_interval = time.time()
                self.phuc.ping_direction('pll')
        elif (x == 1 and y == -1):
            if ((time.time() - self.prl_interval) > 1):
                self.phuc.pinging_RLQ = False
                self.prl_interval = time.time()
                self.phuc.ping_direction('prl')
        #if (self.d_pad_x != x):
        #    print("D pad: X: " + str(x))
        #if (self.d_pad_y != y):
        #    print("D pad: Y: " + str(y))

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

    left_wheel = 90
    right_wheel = 85
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

    yourname_bbc = YourName_BBC()
    #phuc = PHUC_driver.PHUC_driver(left_wheel_neutral=LEFT_WHEEL_NEUTRAL,
    #                               right_wheel_neutral=RIGHT_WHEEL_NEUTRAL,
    #                               left_wheel_acceleration=LEFT_WHEEL_ACCELERATION,
    #                               right_wheel_acceleration=RIGHT_WHEEL_ACCELERATION)
    #xbox_controller = XBoxController(phuc=phuc)
    left_wheel = yourname_bbc.left_wheel_neutral
    right_wheel = yourname_bbc.right_wheel_neutral
    left_wheel_direction = yourname_bbc.left_wheel_neutral
    right_wheel_direction = yourname_bbc.right_wheel_neutral
    green = 0
    red = 0
    blue = 0

    # Set the width and height of the screen (width, height), and name the window.
    screen = pygame.display.set_mode((1100,500))
    pygame.display.set_caption("James_BBC2")

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
        yourname_bbc.phuc.check_channels()
        #if channel.recv_ready():
        #    output = channel.recv(1024).decode('utf-8')
        #    print(output, end='')
        # Event processing step.
        # Possible joystick events: JOYAXISMOTION, JOYBALLMOTION, JOYBUTTONDOWN,
        # JOYBUTTONUP, JOYHATMOTION, JOYDEVICEADDED, JOYDEVICEREMOVED
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False  # Flag that we are done so we exit this loop.

            if event.type == pygame.JOYBUTTONUP or event.type == pygame.JOYBUTTONDOWN:
                yourname_bbc.controller.input(joysticks[0], event)

            # Handle hotplugging
            if event.type == pygame.JOYDEVICEADDED:
                # This event will be generated when the program starts for every
                # joystick, filling up the list without needing to create them manually.
                joy = pygame.joystick.Joystick(event.device_index)
                joysticks[joy.get_instance_id()] = joy
                print(f"Joystick {joy.get_instance_id()} connencted")

            if event.type == pygame.JOYDEVICEREMOVED:
                del joysticks[event.instance_id]
                print(f"Joystick {event.instance_id} disconnected")

        # Drawing step
        # First, clear the screen to white. Don't put other drawing commands
        # above this, or they will be erased with this command.
        screen.fill((255, 255, 255))

        my_surface = pygame.pixelcopy.make_surface(yourname_bbc.phuc.camera_buffer)
        surf = pygame.transform.scale(my_surface, (yourname_bbc.camera_width * 4, yourname_bbc.camera_height * 4))
        screen.blit(surf, (410, 10))

        text_print.reset()

        #Updates program including the controller update function
        yourname_bbc.update(joysticks[0])

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
            text_print.tprint(screen, f"Joystick name: {name}")

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

        # Go ahead and update the screen with what we've drawn.
        pygame.display.flip()

        # Limit to 30 frames per second.
        clock.tick(30)

    print("close phuc")
    yourname_bbc.phuc.close()

if __name__ == '__main__':
    main()
    pygame.quit()
    print("goodbye")