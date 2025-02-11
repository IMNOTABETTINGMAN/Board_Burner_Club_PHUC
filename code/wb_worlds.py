# Created by James J. Smith 12/27/2024
# This file contains the logic for each of the
# webot worlds we create

WEBOTS_WORLDS = ["PHUC_start.wbt","PHUC_balls.wbt","PHUC_balls_drones_wbt","PHUC_line_follower_wbt",
                 "PHUC_real_world","should","be","a","few","more"]

import numpy as np
import random
import math

from controller import supervisor
from controller import InertialUnit

class PHUC_start_wbt:
    def __init__(self, robot, camera_width, camera_height):
        self.robot = robot
        self.camera_width = camera_width
        self.camera_height = camera_height

        # Webots set wheel params
        self.rpm = 100
        self.rps = self.rpm / 60
        self.rad_per_sec = self.rps * 2 * 3.14159
        self.max_speed = self.rad_per_sec

        # reward scale of -1 to 1
        self.reward = [0.0,0.0]
        self.points = 0
        self.record = 0
        self.collision = False
        self.collision_magnitude = 0.0

        self.arena = self.robot.getFromDef('Arena')
        self.arena_floor = self.arena.getField('floorSize')
        self.arena_size = self.arena_floor.getSFVec3f()
        self.arena_wall = self.arena.getField('wallHeight')
        self.arena_wall_height = self.arena_wall.getSFVec3f()
        # print(self.arena_wall_height)

        self.phuc_robot = self.robot.getFromDef('PHUC_robot')
        self.phuc_translation = self.phuc_robot.getField('translation')
        self.phuc_translation_home = self.phuc_translation.getSFVec3f()
        #used in update
        self.phuc_location = self.phuc_translation.getSFVec3f()
        self.phuc_rotation = self.phuc_robot.getField('rotation')
        self.phuc_rotation_home = self.phuc_rotation.getSFVec3f()
        self.left_wheel = self.robot.getDevice('wb_left_wheel')
        self.right_wheel = self.robot.getDevice('wb_right_wheel')
        self.phuc_inertial_unit = InertialUnit('PHUC_inertial_unit')
        self.phuc_inertial_unit.enable(100)
        self.phuc_orientation_pitch = self.phuc_inertial_unit.getRollPitchYaw()[1]
        self.phuc_orientation_yaw = self.phuc_inertial_unit.getRollPitchYaw()[2]
        # print(self.phuc_rotation_home)

        self.phuc_camera = self.robot.getDevice('wb_camera')
        self.camera_fov = self.phuc_camera.getFov()
        # divide fov by 8 individual camera segments representing radians per segment
        self.camera_pixels_per_radian = camera_width/self.camera_fov

        # FOV split into sqares radians per sqare
        self.camera_fov_segment = self.camera_fov/8
        self.camera_pixels_per_segment = self.camera_pixels_per_radian*self.camera_fov_segment
        # print(self.camera_fov)

        # Each segment 20x20 pixels of camera array 8x6 indicated as one hot encoded
        # ball = [1,0,0,0]
        # wall = [0,1,0,0]
        # floor = [0,0,1,0]
        # distance = [0,0,0,1]
        self.camera_flag_key = ['ball','wall','floor','distance']
        # initialize each segment as 'distance
        self.camera_flag_array = [[4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4],
                                  [4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4]]

        self.ping_pong_ball = self.robot.getFromDef('ping_pong_ball_1')
        self.ping_pong_ball_translation = self.ping_pong_ball.getField('translation')
        self.ping_pong_ball_home = self.ping_pong_ball_translation.getSFVec3f()
        self.ping_pong_ball_appearance = self.robot.getFromDef('ppb_appearance')
        self.ping_pong_ball_color = self.ping_pong_ball_appearance.getField('baseColor')
        #print(self.ping_pong_ball_color.getSFVec3f())
        self.ping_pong_ball_colors = [[255/255,255/255,255/255],[14/255,237/255,152/255],[235/255,194/255,12/255],
                                      [46/255,28/255,186/255],[108/255,160/255,224/255],[242/255,167/255,208/255],
                                      [240/255,50/255,154/255],[43/255,153/255,127/255],[245/255, 59/255, 17/255],
                                      [245/255,241/255,10/255]]
        # self.current_color = self.ping_pong_ball_colors[random.randint(0,9)]
        # self.ping_pong_ball_color.setSFVec3f(self.current_color)
        self.current_color = [0.0, 1.0, 0.2]
        self.ping_pong_ball_color.setSFVec3f(self.current_color)

        #distance from ball x direction
        self.adjacent = 0.0
        #distance from ball y direction
        self.opposite = 0.0
        #angle difference between the PHUC camera orientation and the target ball location
        self.angle_difference = 0.0

        # integer values to be placed on camera buffer for display
        self.ball_x_location = 0
        self.ball_y_location = 0

        # creates an array size 100 representing the simplified visual field "state" of the robot
        # -1 maps to 0 representing wall 0 maps to 124 representing floor and 1 maps to 255 representing the target ball
        self.perception_field_len = 100
        self.perception_field = np.zeros(int(self.perception_field_len), dtype=np.float64)
        #pygame surface
        self.perception_field_surface = np.zeros((2,int(self.perception_field_len),3),dtype=np.uint8)

        self.ball_west = 0.0
        self.ball_east = 0.0
        self.ball_north = 0.0
        self.ball_south = 0.0

        self.ticks = 0
    # This is where the supervisor determines the location of all the objects
    # Reward/Punishment functions happen here
    # Relocate objects or reset if necessary
    def update(self):
        self.ticks += 1
        self.collision_magnitude = 0.0

        # first reward is for lw_vel
        # second reward is for rw_vel
        self.reward = [0.0, 0.0]

        # self.ball_west = 0.0
        # self.ball_east = 0.0
        # self.ball_north = 0.0
        # self.ball_south = 0.0

        lw_velocity = self.left_wheel.getVelocity()/self.max_speed
        rw_velocity = self.right_wheel.getVelocity() / self.max_speed
        if lw_velocity > 0.55:
            self.reward[0] += 0.5
        elif abs(lw_velocity) < 0.15:
            self.reward[0] -= 1.0
        elif lw_velocity < -0.5:
            self.reward[0] -= 0.5
        if rw_velocity > 0.55:
            self.reward[1] += 0.5
        elif abs(rw_velocity) < 0.15:
            self.reward[1] -= 1.0
        elif rw_velocity < -0.5:
            self.reward[1] -= 0.5

        self.phuc_location = self.phuc_translation.getSFVec3f()
        self.phuc_orientation_pitch = self.phuc_inertial_unit.getRollPitchYaw()[1]
        self.phuc_orientation_yaw = self.phuc_inertial_unit.getRollPitchYaw()[2]

        ppb_location = self.ping_pong_ball_translation.getSFVec3f()

        # last component corrects for camera offset
        self.adjacent = ppb_location[0] - self.phuc_location[0] - np.cos(self.phuc_orientation_yaw)*0.058
        self.opposite = ppb_location[1] - self.phuc_location[1] - np.sin(self.phuc_orientation_yaw)*0.058
        distance = math.sqrt(self.adjacent*self.adjacent + self.opposite*self.opposite)
        angle = np.arctan(self.opposite/(self.adjacent + 0.0001))
        #second quadrant
        if self.adjacent < 0 < self.opposite:
            angle = 3.1415 + angle
        #third quadrant
        elif self.adjacent < 0 and self.opposite < 0:
            angle = -3.1415 + angle
        # print("adjacent: " + str(adjacent))
        # print("opposite: " + str(opposite))
        # print("ball location: " + str(ppb_location))
        # print("phuc location: " + str(phuc_location))
        # print("phuc orientation: " + str(phuc_orientation))

        # print(self.phuc_orientation_yaw)

        self.angle_difference = self.phuc_orientation_yaw - angle
        if abs(self.angle_difference) < 0.15 and lw_velocity > 0.15 and rw_velocity > 0.15:
            if abs(lw_velocity - rw_velocity) < 0.03:
                self.reward[0] += 10.0
                self.reward[1] += 10.0
            elif abs(lw_velocity - rw_velocity) < 0.08:
                self.reward[0] += 8.0
                self.reward[1] += 8.0
            elif abs(lw_velocity - rw_velocity) < 0.13:
                self.reward[0] += 6.0
                self.reward[1] += 6.0
            elif abs(lw_velocity - rw_velocity) < 0.18:
                self.reward[0] += 4.0
                self.reward[1] += 4.0
            elif abs(lw_velocity - rw_velocity) < 0.23:
                self.reward[0] += 2.0
                self.reward[1] += 2.0
            else:
                self.reward[0] += 1.0
                self.reward[1] += 1.0
            if distance < 0.5:
                self.reward[0] += 2.0
                self.reward[1] += 2.0
            elif distance < 1.5:
                self.reward[0] += 1.5
                self.reward[1] += 1.5
            elif distance < 2.0:
                self.reward[0] += 1.0
                self.reward[1] += 1.0
        elif abs(self.angle_difference) < 0.34 and lw_velocity > 0.15 and rw_velocity > 0.15:
            if abs(lw_velocity - rw_velocity) < 0.03:
                self.reward[0] += 10.0
                self.reward[1] += 10.0
            elif abs(lw_velocity - rw_velocity) < 0.08:
                self.reward[0] += 8.0
                self.reward[1] += 8.0
            elif abs(lw_velocity - rw_velocity) < 0.13:
                self.reward[0] += 6.0
                self.reward[1] += 6.0
            elif abs(lw_velocity - rw_velocity) < 0.18:
                self.reward[0] += 4.0
                self.reward[1] += 4.0
            elif abs(lw_velocity - rw_velocity) < 0.23:
                self.reward[0] += 2.0
                self.reward[1] += 2.0
            else:
                self.reward[0] += 1.0
                self.reward[1] += 1.0
            if distance < 0.5:
                self.reward[0] += 2.0
                self.reward[1] += 2.0
            elif distance < 1.5:
                self.reward[0] += 1.5
                self.reward[1] += 1.5
            elif distance < 2.0:
                self.reward[0] += 1.0
                self.reward[1] += 1.0
        elif abs(self.angle_difference) > 0.40:
            self.reward[0] -= 0.25
            self.reward[1] -= 0.25
        # print("angle diff: " + str(self.angle_difference))

        if abs(self.adjacent) < 0.085 and abs(self.opposite) < 0.085 and abs(self.angle_difference) < 0.40:
            x_rand = random.uniform(-self.arena_size[0] / 2 + 0.03, self.arena_size[0] / 2 - 0.03)
            y_rand = random.uniform(-self.arena_size[1] / 2 + 0.03, self.arena_size[1] / 2 - 0.03)
            self.ping_pong_ball_translation.setSFVec3f([x_rand, y_rand, 0.3])
            # self.current_color = self.ping_pong_ball_colors[random.randint(0, 9)]
            # self.ping_pong_ball_color.setSFVec3f(self.current_color)
            self.ping_pong_ball.resetPhysics()
            self.points += 1
            self.ticks = 0
            self.reward[0] += 25.0
            self.reward[1] += 25.0

        # Wall collision detection
        if self.phuc_location[0] + 0.07 > self.arena_size[0] / 2 or self.phuc_location[0] - 0.07 < -self.arena_size[0] / 2:
            self.collision = True
            # self.collision_magnitude = 0.7
            self.reward[0] -= 20.0
            self.reward[1] -= 20.0
        elif self.phuc_location[0] + 0.12 > self.arena_size[0] / 2 or self.phuc_location[0] - 0.12 < -self.arena_size[0] / 2:
            self.reward[0] -= 1.0
            self.reward[1] -= 1.0
        if self.phuc_location[1] + 0.07 > self.arena_size[1] / 2 or self.phuc_location[1] - 0.07 < -self.arena_size[1] / 2:
            self.collision = True
            # self.collision_magnitude = 0.7
            self.reward[0] -= 20.0
            self.reward[1] -= 20.0
        elif self.phuc_location[1] + 0.12 > self.arena_size[1] / 2 or self.phuc_location[1] - 0.12 < -self.arena_size[1] / 2:
            self.reward[0] -= 1.0
            self.reward[1] -= 1.0

        print("reward: " + str(self.reward))
        #if ppb_location[0] - phuc_location[0] < -0.1:
        #    self.ball_west = 1.0
        #elif ppb_location[0] - phuc_location[0] > 0.1:
        #    self.ball_east = 1.0
        #if ppb_location[1] - phuc_location[1] < -0.1:
        #    self.ball_north = 1.0
        #elif ppb_location[1] - phuc_location[1] > 0.1:
        #    self.ball_north = 1.0
        # print(self.phuc_inertial_unit.getRollPitchYaw())

    def update_perception_field(self, camera_buffer):
        x_center = int(len(camera_buffer)/2)
        y_center = int(len(camera_buffer[0])/2)
        perception_sample_adjust = self.perception_field_len/self.camera_width

        # Sections of camera array to be sent into Conv2D network
        for i in range(-4,4):
            for j in range(-3,3):
                # print(str(x_center) + " " + str(y_center))
                camera_buffer[x_center + i * 20][y_center + j * 20][0] = 255
                camera_buffer[x_center + i * 20][y_center + j * 20][1] = 0
                camera_buffer[x_center + i * 20][y_center + j * 20][2] = 0

                camera_buffer[int(x_center + i*self.camera_pixels_per_segment)][int(y_center + j*self.camera_pixels_per_segment)][0] = 0
                camera_buffer[int(x_center + i*self.camera_pixels_per_segment)][int(y_center + j*self.camera_pixels_per_segment)][1] = 0
                camera_buffer[int(x_center + i*self.camera_pixels_per_segment)][int(y_center + j*self.camera_pixels_per_segment)][2] = 255

        self.ball_x_location = int(x_center + self.angle_difference*self.camera_pixels_per_radian)
        distance_to_ball = math.sqrt(self.opposite*self.opposite + self.adjacent*self.adjacent)
        ball_x_span = int(np.sin(0.02 /distance_to_ball)*self.camera_pixels_per_radian)
        self.ball_y_location = int(
            (np.arctan(-0.0076/distance_to_ball) + self.phuc_orientation_pitch) * self.camera_pixels_per_radian)
        # self.ball_y_location = int((np.arctan(0.02/distance_to_ball)+self.phuc_orientation_pitch)*self.camera_pixels_per_radian)

        #populate array with wall/floor values
        #separate into facing top or bottom wall

        pi = 3.14159
        pi_over_2 = 3.14159/2
        pi_over_4 = 3.14159/4
        pi_3_over_4 = 3.14159*3/4
        distance_top = abs(self.arena_size[1] / 2 - self.phuc_location[1] - np.sin(self.phuc_orientation_yaw) * 0.058)
        distance_bottom = abs(-self.arena_size[1] / 2 - self.phuc_location[1] - np.sin(self.phuc_orientation_yaw) * 0.058)
        distance_left = abs(-self.arena_size[0] / 2 - self.phuc_location[0] - np.cos(self.phuc_orientation_yaw) * 0.058)
        distance_right = abs(self.arena_size[0] / 2 - self.phuc_location[0] - np.cos(self.phuc_orientation_yaw) * 0.058)
        ru_angle = np.arctan(distance_top/distance_right)
        # print("ru: " + str(ru_angle))
        rl_angle = -np.arctan(distance_bottom/distance_right)
        #print("rl: " + str(rl_angle))
        lu_angle = np.arctan(distance_left/distance_top) + pi_over_2
        #print("lu: " + str(lu_angle))
        ll_angle = -np.arctan(distance_left/distance_bottom) - pi_over_2
        #print("ll: " + str(ll_angle))

        # facing top wall
        if ru_angle <= self.phuc_orientation_yaw <= lu_angle:
            for i in range(-50, 50):
                angle = self.phuc_orientation_yaw - i/self.camera_pixels_per_radian
                if angle <= ru_angle:
                    distance_to_wall = distance_right / np.cos(angle)
                elif angle >= lu_angle:
                    angle = abs(angle - pi)
                    distance_to_wall = distance_left / np.cos(angle)
                else:
                    angle = abs(angle - pi_over_2)
                    distance_to_wall = distance_top/np.cos(angle)
                self.perception_field[50 + i] = -np.exp(-(distance_to_wall*1.8))
        # facing bottom wall
        elif ll_angle <= self.phuc_orientation_yaw <= rl_angle:
            for i in range(-50, 50):
                angle = self.phuc_orientation_yaw - i / self.camera_pixels_per_radian
                if angle <= ll_angle:
                    angle = abs(angle + pi)
                    distance_to_wall = distance_left / np.cos(angle)
                elif angle >= rl_angle:
                    angle = abs(angle)
                    distance_to_wall = distance_right / np.cos(angle)
                else:
                    angle = abs(angle + pi_over_2)
                    distance_to_wall = distance_bottom / np.cos(angle)
                self.perception_field[50 + i] = -np.exp(-(distance_to_wall*1.8))
        # facing right wall
        elif rl_angle <= self.phuc_orientation_yaw <= ru_angle:
            for i in range(-50, 50):
                angle = self.phuc_orientation_yaw - i / self.camera_pixels_per_radian
                if angle >= ru_angle:
                    angle = abs(angle - pi_over_2)
                    distance_to_wall = distance_top / np.cos(angle)
                elif angle <= rl_angle:
                    angle = abs(angle + pi_over_2)
                    distance_to_wall = distance_bottom / np.cos(angle)
                else:
                    angle = abs(angle)
                    distance_to_wall = distance_right / np.cos(angle)
                self.perception_field[50 + i] = -np.exp(-(distance_to_wall*1.8))
        # facing left wall
        else:
            if self.phuc_orientation_yaw > 0:
                lw_angle = pi - self.phuc_orientation_yaw
            else:
                lw_angle = -(pi + self.phuc_orientation_yaw)
            for i in range(-50, 50):
                angle = lw_angle + i / self.camera_pixels_per_radian
                if abs(ll_angle) - pi >= angle:
                    angle = abs(angle + pi_over_2)
                    distance_to_wall = distance_bottom / np.cos(angle)
                elif abs(ll_angle) - pi <= angle <= pi - lu_angle:
                    angle = abs(angle)
                    distance_to_wall = distance_left / np.cos(angle)
                else:
                    angle = abs(angle - pi_over_2)
                    distance_to_wall = distance_top / np.cos(angle)
                self.perception_field[50 + i] = -np.exp(-(distance_to_wall*1.8))

        #for i in range(self.perception_field_len):
        #    self.perception_field[i] = -0.5
        if 0 <= self.ball_x_location <= len(camera_buffer) - 1:
            camera_buffer[self.ball_x_location][int(y_center - self.ball_y_location)][0] = 0
            camera_buffer[self.ball_x_location][int(y_center - self.ball_y_location)][1] = 0
            camera_buffer[self.ball_x_location][int(y_center - self.ball_y_location)][2] = 0
        for i in range(-ball_x_span,ball_x_span+1):
            if 0 <= self.ball_x_location + i <= len(camera_buffer)-1:
                self.perception_field[int((self.ball_x_location + i) * perception_sample_adjust)] = 1
                self.perception_field[int((self.ball_x_location + i) * perception_sample_adjust)] = 1
                # camera_buffer[self.ball_x_location + i][int(y_center - self.ball_y_location)][0] = 0
                # camera_buffer[self.ball_x_location + i][int(y_center - self.ball_y_location)][1] = 0
                # camera_buffer[self.ball_x_location + i][int(y_center - self.ball_y_location)][2] = 0

        for i in range(self.perception_field_len):
            self.perception_field_surface[0][i][0] = int((self.perception_field[i]*0.5 + 0.5)*255)
            self.perception_field_surface[0][i][1] = int((self.perception_field[i]*0.5 + 0.5)*255)
            self.perception_field_surface[0][i][2] = int((self.perception_field[i]*0.5 + 0.5)*255)
            self.perception_field_surface[1][i][0] = int((self.perception_field[i]*0.5 + 0.5)*255)
            self.perception_field_surface[1][i][1] = int((self.perception_field[i]*0.5 + 0.5)*255)
            self.perception_field_surface[1][i][2] = int((self.perception_field[i]*0.5 + 0.5)*255)

        # print(self.perception_field)
        return camera_buffer

    def determine_reward(self):
        return self.reward

    def episode_finished(self):
        if self.collision or self.points > 100 or self.ticks > 3000:
            if self.points > self.record:
                self.record = self.points
            self.ticks = 0
            return 1.0
        else:
            return 0.0

    def reset(self):
        self.reward = 0
        self.points = 0
        self.collision = False
        self.phuc_translation.setSFVec3f(self.phuc_translation_home)
        self.phuc_rotation.setSFRotation(self.phuc_rotation_home)
        self.phuc_robot.resetPhysics()

        self.ping_pong_ball_translation.setSFVec3f(self.ping_pong_ball_home)
        self.ping_pong_ball.resetPhysics()

class PHUC_balls_wbt:
    def __init__(self, robot):
        self.robot = robot

        # Webots set wheel params
        self.rpm = 100
        self.rps = self.rpm / 60
        self.rad_per_sec = self.rps * 2 * 3.14159
        self.max_speed = self.rad_per_sec

        # reward scale of -1 to 1
        self.reward = [0.0,0.0]
        self.points = 0
        self.record = 0
        self.collision = False
        self.collision_magnitude = 0.0

        self.arena = self.robot.getFromDef('Arena')
        print(self.arena)
        self.arena_floor = self.arena.getField('floorSize')
        self.arena_size = self.arena_floor.getSFVec3f()
        # print(self.arena_size)

        self.phuc_robot = self.robot.getFromDef('drone_1')
        self.phuc_translation = self.phuc_robot.getField('translation')
        self.phuc_translation_home = self.phuc_translation.getSFVec3f()
        self.phuc_rotation = self.phuc_robot.getField('rotation')
        self.phuc_rotation_home = self.phuc_rotation.getSFVec3f()
        self.left_wheel = self.robot.getDevice('wb_left_wheel')
        self.right_wheel = self.robot.getDevice('wb_right_wheel')
        self.phuc_inertial_unit = InertialUnit('drone_1_inertial_unit')
        self.phuc_inertial_unit.enable(100)
        # print(self.phuc_rotation_home)

        self.ping_pong_ball = self.robot.getFromDef('ping_pong_ball_goal')
        self.ping_pong_ball_translation = self.ping_pong_ball.getField('translation')
        self.ping_pong_ball_home = self.ping_pong_ball_translation.getSFVec3f()
        self.ping_pong_ball_appearance = self.robot.getFromDef('ppb_appearance')
        self.ping_pong_ball_color = self.ping_pong_ball_appearance.getField('baseColor')
        #print(self.ping_pong_ball_color.getSFVec3f())
        self.ping_pong_ball_colors = [[255/255,255/255,255/255],[14/255,237/255,152/255],[235/255,194/255,12/255],
                                      [46/255,28/255,186/255],[108/255,160/255,224/255],[242/255,167/255,208/255],
                                      [240/255,50/255,154/255],[43/255,153/255,127/255],[245/255, 59/255, 17/255],
                                      [245/255,241/255,10/255]]
        # self.current_color = self.ping_pong_ball_colors[random.randint(0,9)]
        # self.ping_pong_ball_color.setSFVec3f(self.current_color)
        self.current_color = [0.0, 1.0, 0.2]
        self.ping_pong_ball_color.setSFVec3f(self.current_color)

        self.ball_1 = self.robot.getFromDef('ball_1')
        self.ball_1_translation = self.ping_pong_ball.getField('translation')
        self.ball_1_home = self.ping_pong_ball_translation.getSFVec3f()
        self.ball_1_theta = random.uniform(0, 2*3.14159)
        self.ball_1_velocity = random.uniform(0.1,1)
        self.ball_1_vel_y = math.sin(self.ball_1_theta)*self.ball_1_velocity
        self.ball_1_vel_x = math.cos(self.ball_1_theta) * self.ball_1_velocity

        self.ball_2 = self.robot.getFromDef('ball_1')
        self.ball_2_translation = self.ping_pong_ball.getField('translation')
        self.ball_2_home = self.ping_pong_ball_translation.getSFVec3f()
        self.ball_2_theta = random.uniform(0, 2 * 3.14159)
        self.ball_2_velocity = random.uniform(0.1, 1)
        self.ball_2_vel_y = math.sin(self.ball_2_theta) * self.ball_2_velocity
        self.ball_2_vel_x = math.cos(self.ball_2_theta) * self.ball_2_velocity

        self.ball_3 = self.robot.getFromDef('ball_1')
        self.ball_3_translation = self.ping_pong_ball.getField('translation')
        self.ball_3_home = self.ping_pong_ball_translation.getSFVec3f()
        self.ball_3_theta = random.uniform(0, 2 * 3.14159)
        self.ball_3_velocity = random.uniform(0.1, 1)
        self.ball_3_vel_y = math.sin(self.ball_3_theta) * self.ball_3_velocity
        self.ball_3_vel_x = math.cos(self.ball_3_theta) * self.ball_3_velocity

        self.ball_4 = self.robot.getFromDef('ball_1')
        self.ball_4_translation = self.ping_pong_ball.getField('translation')
        self.ball_4_home = self.ping_pong_ball_translation.getSFVec3f()
        self.ball_4_theta = random.uniform(0, 2 * 3.14159)
        self.ball_4_velocity = random.uniform(0.1, 1)
        self.ball_4_vel_y = math.sin(self.ball_4_theta) * self.ball_4_velocity
        self.ball_4_vel_x = math.cos(self.ball_4_theta) * self.ball_4_velocity

        self.ball_west = 0.0
        self.ball_east = 0.0
        self.ball_north = 0.0
        self.ball_south = 0.0

    # This is where the supervisor determines the location of all the objects
    # Reward/Punishment functions happen here
    # Relocate objects or reset if necessary
    def update(self):
        self.ball_west = 0.0
        self.ball_east = 0.0
        self.ball_north = 0.0
        self.ball_south = 0.0
        self.collision_magnitude = 0.0

        # first reward is for movement
        # second reward is for light display
        self.reward = [0.0,0.0]


        lw_velocity = self.left_wheel.getVelocity()/self.max_speed
        rw_velocity = self.right_wheel.getVelocity() / self.max_speed
        if lw_velocity > 0.65:
            self.reward[0] += 5.0
        elif abs(lw_velocity) < 0.15:
            self.reward[0] -= 1.0
        elif lw_velocity < -0.5:
            self.reward[0] -= 1.0
        if rw_velocity > 0.65:
            self.reward[1] += 5.0
        elif abs(rw_velocity) < 0.15:
            self.reward[1] -= 1.0
        elif rw_velocity < -0.5:
            self.reward[1] -= 1.0

        phuc_location = self.phuc_translation.getSFVec3f()
        phuc_orientation = self.phuc_inertial_unit.getRollPitchYaw()[2]
        ppb_location = self.ping_pong_ball_translation.getSFVec3f()

        adjacent = ppb_location[0] - phuc_location[0]
        opposite = ppb_location[1] - phuc_location[1]
        angle = np.arctan(opposite/(adjacent + 0.0001))
        #second quadrant
        if adjacent < 0 < opposite:
            angle = 3.1415 + angle
        #third quadrant
        elif adjacent < 0 and opposite < 0:
            angle = -3.1415 + angle
        # print("adjacent: " + str(adjacent))
        # print("opposite: " + str(opposite))
        # print("ball location: " + str(ppb_location))
        # print("phuc location: " + str(phuc_location))
        #print("phuc orientation: " + str(phuc_orientation))


        angle_diff = abs(phuc_orientation - angle)
        if angle_diff < 0.1 and lw_velocity > 0.15 and rw_velocity > 0.15 and abs(lw_velocity - rw_velocity) < 0.1:
            self.reward[0] += 10.0
            self.reward[1] += 10.0
            if abs(adjacent) < 0.2 and abs(opposite) < 0.2:
                self.reward[0] += 8.0
                self.reward[1] += 8.0
            elif abs(adjacent) < 0.3 and abs(opposite) < 0.3:
                self.reward[0] += 6.0
                self.reward[1] += 6.0
            elif abs(adjacent) < 0.4 and abs(opposite) < 0.4:
                self.reward[0] += 4.0
                self.reward[1] += 4.0
        elif angle_diff < 0.36 and lw_velocity > 0.15 and rw_velocity > 0.15 and abs(lw_velocity - rw_velocity) < 0.1:
            self.reward[0] += 8.0
            self.reward[1] += 8.0
            if abs(adjacent) < 0.2 and abs(opposite) < 0.2:
                self.reward[0] += 8.0
                self.reward[1] += 8.0
            elif abs(adjacent) < 0.3 and abs(opposite) < 0.3:
                self.reward[0] += 6.0
                self.reward[1] += 6.0
            elif abs(adjacent) < 0.4 and abs(opposite) < 0.4:
                self.reward[0] += 4.0
                self.reward[1] += 4.0
        # print("angle diff: " + str(angle_diff))

        if abs(adjacent) < 0.100 and abs(opposite) < 0.100 and angle_diff < 0.45:
            x_rand = random.uniform(-self.arena_size[0] / 2 + 0.03, self.arena_size[0] / 2 - 0.03)
            y_rand = random.uniform(-self.arena_size[1] / 2 + 0.03, self.arena_size[1] / 2 - 0.03)
            self.ping_pong_ball_translation.setSFVec3f([x_rand, y_rand, 0.3])
            # self.current_color = self.ping_pong_ball_colors[random.randint(0, 9)]
            # self.ping_pong_ball_color.setSFVec3f(self.current_color)
            self.ping_pong_ball.resetPhysics()
            self.points += 1
            self.reward[0] += 40.0
            self.reward[1] += 40.0

        # Wall collision detection
        if phuc_location[0] + 0.07 > self.arena_size[0] / 2 or phuc_location[0] - 0.07 < -self.arena_size[0] / 2:
            self.collision = True
            self.collision_magnitude = 1.0
            self.reward[0] -= 20.0
            self.reward[1] -= 20.0
        elif phuc_location[0] + 0.12 > self.arena_size[0] / 2 or phuc_location[0] - 0.12 < -self.arena_size[0] / 2:
            self.reward[0] -= 3.0
            self.reward[1] -= 3.0
        if phuc_location[1] + 0.07 > self.arena_size[1] / 2 or phuc_location[1] - 0.07 < -self.arena_size[1] / 2:
            self.collision = True
            self.collision_magnitude = 1.0
            self.reward[0] -= 20.0
            self.reward[1] -= 20.0
        elif phuc_location[1] + 0.12 > self.arena_size[1] / 2 or phuc_location[1] - 0.12 < -self.arena_size[1] / 2:
            self.reward[0] -= 3.0
            self.reward[1] -= 3.0

        # print(self.reward)
        #if ppb_location[0] - phuc_location[0] < -0.1:
        #    self.ball_west = 1.0
        #elif ppb_location[0] - phuc_location[0] > 0.1:
        #    self.ball_east = 1.0
        #if ppb_location[1] - phuc_location[1] < -0.1:
        #    self.ball_north = 1.0
        #elif ppb_location[1] - phuc_location[1] > 0.1:
        #    self.ball_north = 1.0
        # print(self.phuc_inertial_unit.getRollPitchYaw())

    def determine_reward(self):
        return self.reward

    def episode_finished(self):
        if self.collision or self.points > 100:
            if self.points > self.record:
                self.record = self.points
            return 1.0
        else:
            return 0.0

    def reset(self):
        self.reward = 0
        self.points = 0
        self.collision = False
        self.collision_magnitude = 0.0
        self.phuc_translation.setSFVec3f(self.phuc_translation_home)
        self.phuc_rotation.setSFRotation(self.phuc_rotation_home)
        self.phuc_robot.resetPhysics()

        self.ping_pong_ball_translation.setSFVec3f(self.ping_pong_ball_home)
        self.ping_pong_ball.resetPhysics()

        self.ball_1_translation.setSFVec3f(self.ball_1_home)
        self.ball_1.resetPhysics()
        self.ball_1_theta = random.uniform(0, 2 * 3.14159)
        self.ball_1_velocity = random.uniform(0.1, 1)
        self.ball_1_vel_y = math.sin(self.ball_1_theta) * self.ball_1_velocity
        self.ball_1_vel_x = math.cos(self.ball_1_theta) * self.ball_1_velocity

        self.ball_2_translation.setSFVec3f(self.ball_2_home)
        self.ball_2.resetPhysics()
        self.ball_2_theta = random.uniform(0, 2 * 3.14159)
        self.ball_2_velocity = random.uniform(0.1, 1)
        self.ball_2_vel_y = math.sin(self.ball_2_theta) * self.ball_2_velocity
        self.ball_2_vel_x = math.cos(self.ball_2_theta) * self.ball_2_velocity

        self.ball_3_translation.setSFVec3f(self.ball_3_home)
        self.ball_3.resetPhysics()
        self.ball_3_theta = random.uniform(0, 2 * 3.14159)
        self.ball_3_velocity = random.uniform(0.1, 1)
        self.ball_3_vel_y = math.sin(self.ball_3_theta) * self.ball_3_velocity
        self.ball_3_vel_x = math.cos(self.ball_3_theta) * self.ball_3_velocity

        self.ball_4_translation.setSFVec3f(self.ball_4_home)
        self.ball_4.resetPhysics()
        self.ball_4_theta = random.uniform(0, 2 * 3.14159)
        self.ball_4_velocity = random.uniform(0.1, 1)
        self.ball_4_vel_y = math.sin(self.ball_4_theta) * self.ball_4_velocity
        self.ball_4_vel_x = math.cos(self.ball_4_theta) * self.ball_4_velocity

class PHUC_balls_drones_wbt:
    def __init__(self, robot, camera_width, camera_height):
        self.robot = robot
        self.camera_width = camera_width
        self.camera_height = camera_height

        # Webots set wheel params
        self.rpm = 100
        self.rps = self.rpm / 60
        self.rad_per_sec = self.rps * 2 * 3.14159
        self.max_speed = self.rad_per_sec

        # reward scale of -1 to 1
        self.reward = [0.0,0.0]
        self.points = 0
        self.record = 0
        self.collision = False
        self.collision_magnitude = 0.0

        self.arena = self.robot.getFromDef('Arena')
        self.arena_floor = self.arena.getField('floorSize')
        self.arena_size = self.arena_floor.getSFVec3f()
        self.arena_wall = self.arena.getField('wallHeight')
        self.arena_wall_height = self.arena_wall.getSFVec3f()
        # print(self.arena_wall_height)

        self.phuc_robot = self.robot.getFromDef('PHUC_robot')
        self.phuc_translation = self.phuc_robot.getField('translation')
        self.phuc_translation_home = self.phuc_translation.getSFVec3f()
        #used in update
        self.phuc_location = self.phuc_translation.getSFVec3f()
        self.phuc_rotation = self.phuc_robot.getField('rotation')
        self.phuc_rotation_home = self.phuc_rotation.getSFVec3f()
        self.left_wheel = self.robot.getDevice('wb_left_wheel')
        self.right_wheel = self.robot.getDevice('wb_right_wheel')
        self.phuc_inertial_unit = InertialUnit('PHUC_inertial_unit')
        self.phuc_inertial_unit.enable(100)
        self.phuc_orientation_pitch = self.phuc_inertial_unit.getRollPitchYaw()[1]
        self.phuc_orientation_yaw = self.phuc_inertial_unit.getRollPitchYaw()[2]
        # print(self.phuc_rotation_home)

        self.drone_robot = self.robot.getFromDef('drone')
        self.drone_translation = self.phuc_robot.getField('translation')
        self.drone_translation_home = self.phuc_translation.getSFVec3f()
        # used in update
        self.drone_location = self.phuc_translation.getSFVec3f()
        self.drone_rotation = self.phuc_robot.getField('rotation')
        self.drone_rotation_home = self.phuc_rotation.getSFVec3f()
        self.drone_left_wheel = self.robot.getDevice('wb_left_wheel')
        self.drone_right_wheel = self.robot.getDevice('wb_right_wheel')
        self.drone_inertial_unit = InertialUnit('PHUC_inertial_unit')
        self.drone_inertial_unit.enable(100)
        self.drone_orientation_pitch = self.phuc_inertial_unit.getRollPitchYaw()[1]
        self.drone_orientation_yaw = self.phuc_inertial_unit.getRollPitchYaw()[2]
        # print(self.phuc_rotation_home)

        self.phuc_camera = self.robot.getDevice('wb_camera')
        self.camera_fov = self.phuc_camera.getFov()
        # divide fov by 8 individual camera segments representing radians per segment
        self.camera_pixels_per_radian = camera_width/self.camera_fov

        # FOV split into sqares radians per sqare
        self.camera_fov_segment = self.camera_fov/8
        self.camera_pixels_per_segment = self.camera_pixels_per_radian*self.camera_fov_segment
        # print(self.camera_fov)

        # Each segment 20x20 pixels of camera array 8x6 indicated as one hot encoded
        # ball = [1,0,0,0]
        # wall = [0,1,0,0]
        # floor = [0,0,1,0]
        # distance = [0,0,0,1]
        self.camera_flag_key = ['ball','wall','floor','distance']
        # initialize each segment as 'distance
        self.camera_flag_array = [[4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4],
                                  [4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4]]

        # PHUC ball
        self.ping_pong_ball = self.robot.getFromDef('ping_pong_ball_1')
        self.ping_pong_ball_translation = self.ping_pong_ball.getField('translation')
        self.ping_pong_ball_home = self.ping_pong_ball_translation.getSFVec3f()
        self.ping_pong_ball_appearance = self.robot.getFromDef('ppb_appearance')
        self.ping_pong_ball_color = self.ping_pong_ball_appearance.getField('baseColor')
        #print(self.ping_pong_ball_color.getSFVec3f())
        self.ping_pong_ball_colors = [[255/255,255/255,255/255],[14/255,237/255,152/255],[235/255,194/255,12/255],
                                      [46/255,28/255,186/255],[108/255,160/255,224/255],[242/255,167/255,208/255],
                                      [240/255,50/255,154/255],[43/255,153/255,127/255],[245/255, 59/255, 17/255],
                                      [245/255,241/255,10/255]]
        # self.current_color = self.ping_pong_ball_colors[random.randint(0,9)]
        # self.ping_pong_ball_color.setSFVec3f(self.current_color)
        self.current_color = [0.0, 1.0, 0.2]
        self.ping_pong_ball_color.setSFVec3f(self.current_color)

        # Drone ball
        self.drone_ball = self.robot.getFromDef('drone_ball')
        self.drone_ball_translation = self.ping_pong_ball.getField('translation')
        self.drone_ball_home = self.ping_pong_ball_translation.getSFVec3f()
        self.drone_ball_appearance = self.robot.getFromDef('ppb_appearance')
        self.drone_ball_color = self.ping_pong_ball_appearance.getField('baseColor')
        # self.current_color = self.ping_pong_ball_colors[random.randint(0,9)]
        # self.ping_pong_ball_color.setSFVec3f(self.current_color)
        self.drone_current_color = [0.0, 0.3, 1.0]
        self.drone_ball_color.setSFVec3f(self.current_color)

        #distance from ball x direction
        self.adjacent = 0.0
        #distance from ball y direction
        self.opposite = 0.0
        #angle difference between the PHUC camera orientation and the target ball location
        self.angle_difference = 0.0

        # integer values to be placed on camera buffer for display
        self.ball_x_location = 0
        self.ball_y_location = 0

        # creates an array size 100 representing the simplified visual field "state" of the robot
        # -1 maps to 0 representing wall 0 maps to 124 representing floor and 1 maps to 255 representing the target ball
        self.perception_field_len = 100
        self.perception_field = np.zeros(int(self.perception_field_len), dtype=np.float64)
        #pygame surface
        self.perception_field_surface = np.zeros((2,int(self.perception_field_len),3),dtype=np.uint8)

        self.ball_west = 0.0
        self.ball_east = 0.0
        self.ball_north = 0.0
        self.ball_south = 0.0

        self.ticks = 0
    # This is where the supervisor determines the location of all the objects
    # Reward/Punishment functions happen here
    # Relocate objects or reset if necessary
    def update(self):
        self.ticks += 1
        self.collision_magnitude = 0.0

        # first reward is for lw_vel
        # second reward is for rw_vel
        self.reward = [0.0, 0.0]

        # self.ball_west = 0.0
        # self.ball_east = 0.0
        # self.ball_north = 0.0
        # self.ball_south = 0.0

        lw_velocity = self.left_wheel.getVelocity()/self.max_speed
        rw_velocity = self.right_wheel.getVelocity() / self.max_speed
        if lw_velocity > 0.55:
            self.reward[0] += 0.5
        elif abs(lw_velocity) < 0.15:
            self.reward[0] -= 1.0
        elif lw_velocity < -0.5:
            self.reward[0] -= 0.5
        if rw_velocity > 0.55:
            self.reward[1] += 0.5
        elif abs(rw_velocity) < 0.15:
            self.reward[1] -= 1.0
        elif rw_velocity < -0.5:
            self.reward[1] -= 0.5

        self.phuc_location = self.phuc_translation.getSFVec3f()
        self.phuc_orientation_pitch = self.phuc_inertial_unit.getRollPitchYaw()[1]
        self.phuc_orientation_yaw = self.phuc_inertial_unit.getRollPitchYaw()[2]

        ppb_location = self.ping_pong_ball_translation.getSFVec3f()

        # last component corrects for camera offset
        self.adjacent = ppb_location[0] - self.phuc_location[0] - np.cos(self.phuc_orientation_yaw)*0.058
        self.opposite = ppb_location[1] - self.phuc_location[1] - np.sin(self.phuc_orientation_yaw)*0.058
        angle = np.arctan(self.opposite/(self.adjacent + 0.0001))
        #second quadrant
        if self.adjacent < 0 < self.opposite:
            angle = 3.1415 + angle
        #third quadrant
        elif self.adjacent < 0 and self.opposite < 0:
            angle = -3.1415 + angle
        # print("adjacent: " + str(adjacent))
        # print("opposite: " + str(opposite))
        # print("ball location: " + str(ppb_location))
        # print("phuc location: " + str(phuc_location))
        # print("phuc orientation: " + str(phuc_orientation))

        # print(self.phuc_orientation_yaw)

        self.angle_difference = self.phuc_orientation_yaw - angle
        if abs(self.angle_difference) < 0.15 and lw_velocity > 0.15 and rw_velocity > 0.15 and abs(lw_velocity - rw_velocity) < 0.20:
            self.reward[0] += 6.0
            self.reward[1] += 6.0
            if abs(self.adjacent) < 0.3 and abs(self.opposite) < 0.3:
                self.reward[0] += 2.0
                self.reward[1] += 2.0
            elif abs(self.adjacent) < 0.8 and abs(self.opposite) < 0.8:
                self.reward[0] += 1.5
                self.reward[1] += 1.5
            elif abs(self.adjacent) < 1.5 and abs(self.opposite) < 1.5:
                self.reward[0] += 1.0
                self.reward[1] += 1.0
        elif abs(self.angle_difference) < 0.30 and lw_velocity > 0.15 and rw_velocity > 0.15 and abs(lw_velocity - rw_velocity) < 0.20:
            self.reward[0] += 4.0
            self.reward[1] += 4.0
            if abs(self.adjacent) < 0.3 and abs(self.opposite) < 0.3:
                self.reward[0] += 2.0
                self.reward[1] += 2.0
            elif abs(self.adjacent) < 0.8 and abs(self.opposite) < 0.8:
                self.reward[0] += 1.5
                self.reward[1] += 1.5
            elif abs(self.adjacent) < 1.5 and abs(self.opposite) < 1.5:
                self.reward[0] += 1.0
                self.reward[1] += 1.0
        # print("angle diff: " + str(self.angle_difference))

        if abs(self.adjacent) < 0.100 and abs(self.opposite) < 0.100 and abs(self.angle_difference) < 0.45:
            x_rand = random.uniform(-self.arena_size[0] / 2 + 0.03, self.arena_size[0] / 2 - 0.03)
            y_rand = random.uniform(-self.arena_size[1] / 2 + 0.03, self.arena_size[1] / 2 - 0.03)
            self.ping_pong_ball_translation.setSFVec3f([x_rand, y_rand, 0.3])
            # self.current_color = self.ping_pong_ball_colors[random.randint(0, 9)]
            # self.ping_pong_ball_color.setSFVec3f(self.current_color)
            self.ping_pong_ball.resetPhysics()
            self.points += 1
            self.ticks = 0
            self.reward[0] += 20.0
            self.reward[1] += 20.0

        # Wall collision detection
        if self.phuc_location[0] + 0.07 > self.arena_size[0] / 2 or self.phuc_location[0] - 0.07 < -self.arena_size[0] / 2:
            self.collision = True
            self.collision_magnitude = 0.7
            self.reward[0] -= 20.0
            self.reward[1] -= 20.0
        elif self.phuc_location[0] + 0.12 > self.arena_size[0] / 2 or self.phuc_location[0] - 0.12 < -self.arena_size[0] / 2:
            self.reward[0] -= 1.0
            self.reward[1] -= 1.0
        if self.phuc_location[1] + 0.07 > self.arena_size[1] / 2 or self.phuc_location[1] - 0.07 < -self.arena_size[1] / 2:
            self.collision = True
            self.collision_magnitude = 0.7
            self.reward[0] -= 20.0
            self.reward[1] -= 20.0
        elif self.phuc_location[1] + 0.12 > self.arena_size[1] / 2 or self.phuc_location[1] - 0.12 < -self.arena_size[1] / 2:
            self.reward[0] -= 1.0
            self.reward[1] -= 1.0

        # print(self.reward)
        #if ppb_location[0] - phuc_location[0] < -0.1:
        #    self.ball_west = 1.0
        #elif ppb_location[0] - phuc_location[0] > 0.1:
        #    self.ball_east = 1.0
        #if ppb_location[1] - phuc_location[1] < -0.1:
        #    self.ball_north = 1.0
        #elif ppb_location[1] - phuc_location[1] > 0.1:
        #    self.ball_north = 1.0
        # print(self.phuc_inertial_unit.getRollPitchYaw())

    def update_perception_field(self, camera_buffer):
        x_center = int(len(camera_buffer)/2)
        y_center = int(len(camera_buffer[0])/2)
        perception_sample_adjust = self.perception_field_len/self.camera_width

        # Sections of camera array to be sent into Conv2D network
        for i in range(-4,4):
            for j in range(-3,3):
                # print(str(x_center) + " " + str(y_center))
                camera_buffer[x_center + i * 20][y_center + j * 20][0] = 255
                camera_buffer[x_center + i * 20][y_center + j * 20][1] = 0
                camera_buffer[x_center + i * 20][y_center + j * 20][2] = 0

                camera_buffer[int(x_center + i*self.camera_pixels_per_segment)][int(y_center + j*self.camera_pixels_per_segment)][0] = 0
                camera_buffer[int(x_center + i*self.camera_pixels_per_segment)][int(y_center + j*self.camera_pixels_per_segment)][1] = 0
                camera_buffer[int(x_center + i*self.camera_pixels_per_segment)][int(y_center + j*self.camera_pixels_per_segment)][2] = 255

        self.ball_x_location = int(x_center + self.angle_difference*self.camera_pixels_per_radian)
        distance_to_ball = math.sqrt(self.opposite*self.opposite + self.adjacent*self.adjacent)
        ball_x_span = int(np.sin(0.02 /distance_to_ball)*self.camera_pixels_per_radian)
        self.ball_y_location = int(
            (np.arctan(-0.0076/distance_to_ball) + self.phuc_orientation_pitch) * self.camera_pixels_per_radian)
        # self.ball_y_location = int((np.arctan(0.02/distance_to_ball)+self.phuc_orientation_pitch)*self.camera_pixels_per_radian)

        #populate array with wall/floor values
        #separate into facing top or bottom wall

        pi = 3.14159
        pi_over_2 = 3.14159/2
        pi_over_4 = 3.14159/4
        pi_3_over_4 = 3.14159*3/4
        distance_top = abs(self.arena_size[1] / 2 - self.phuc_location[1] - np.sin(self.phuc_orientation_yaw) * 0.058)
        distance_bottom = abs(-self.arena_size[1] / 2 - self.phuc_location[1] - np.sin(self.phuc_orientation_yaw) * 0.058)
        distance_left = abs(-self.arena_size[0] / 2 - self.phuc_location[0] - np.cos(self.phuc_orientation_yaw) * 0.058)
        distance_right = abs(self.arena_size[0] / 2 - self.phuc_location[0] - np.cos(self.phuc_orientation_yaw) * 0.058)
        ru_angle = np.arctan(distance_top/distance_right)
        # print("ru: " + str(ru_angle))
        rl_angle = -np.arctan(distance_bottom/distance_right)
        #print("rl: " + str(rl_angle))
        lu_angle = np.arctan(distance_left/distance_top) + pi_over_2
        #print("lu: " + str(lu_angle))
        ll_angle = -np.arctan(distance_left/distance_bottom) - pi_over_2
        #print("ll: " + str(ll_angle))

        # facing top wall
        if ru_angle <= self.phuc_orientation_yaw <= lu_angle:
            for i in range(-50, 50):
                angle = self.phuc_orientation_yaw - i/self.camera_pixels_per_radian
                if angle <= ru_angle:
                    distance_to_wall = distance_right / np.cos(angle)
                elif angle >= lu_angle:
                    angle = abs(angle - pi)
                    distance_to_wall = distance_left / np.cos(angle)
                else:
                    angle = abs(angle - pi_over_2)
                    distance_to_wall = distance_top/np.cos(angle)
                self.perception_field[50 + i] = -np.exp(-(distance_to_wall*1.8))
        # facing bottom wall
        elif ll_angle <= self.phuc_orientation_yaw <= rl_angle:
            for i in range(-50, 50):
                angle = self.phuc_orientation_yaw - i / self.camera_pixels_per_radian
                if angle <= ll_angle:
                    angle = abs(angle + pi)
                    distance_to_wall = distance_left / np.cos(angle)
                elif angle >= rl_angle:
                    angle = abs(angle)
                    distance_to_wall = distance_right / np.cos(angle)
                else:
                    angle = abs(angle + pi_over_2)
                    distance_to_wall = distance_bottom / np.cos(angle)
                self.perception_field[50 + i] = -np.exp(-(distance_to_wall*1.8))
        # facing right wall
        elif rl_angle <= self.phuc_orientation_yaw <= ru_angle:
            for i in range(-50, 50):
                angle = self.phuc_orientation_yaw - i / self.camera_pixels_per_radian
                if angle >= ru_angle:
                    angle = abs(angle - pi_over_2)
                    distance_to_wall = distance_top / np.cos(angle)
                elif angle <= rl_angle:
                    angle = abs(angle + pi_over_2)
                    distance_to_wall = distance_bottom / np.cos(angle)
                else:
                    angle = abs(angle)
                    distance_to_wall = distance_right / np.cos(angle)
                self.perception_field[50 + i] = -np.exp(-(distance_to_wall*1.8))
        # facing left wall
        else:
            if self.phuc_orientation_yaw > 0:
                lw_angle = pi - self.phuc_orientation_yaw
            else:
                lw_angle = -(pi + self.phuc_orientation_yaw)
            for i in range(-50, 50):
                angle = lw_angle + i / self.camera_pixels_per_radian
                if abs(ll_angle) - pi >= angle:
                    angle = abs(angle + pi_over_2)
                    distance_to_wall = distance_bottom / np.cos(angle)
                elif abs(ll_angle) - pi <= angle <= pi - lu_angle:
                    angle = abs(angle)
                    distance_to_wall = distance_left / np.cos(angle)
                else:
                    angle = abs(angle - pi_over_2)
                    distance_to_wall = distance_top / np.cos(angle)
                self.perception_field[50 + i] = -np.exp(-(distance_to_wall*1.8))

        #for i in range(self.perception_field_len):
        #    self.perception_field[i] = -0.5
        if 0 <= self.ball_x_location <= len(camera_buffer) - 1:
            camera_buffer[self.ball_x_location][int(y_center - self.ball_y_location)][0] = 0
            camera_buffer[self.ball_x_location][int(y_center - self.ball_y_location)][1] = 0
            camera_buffer[self.ball_x_location][int(y_center - self.ball_y_location)][2] = 0
        for i in range(-ball_x_span,ball_x_span+1):
            if 0 <= self.ball_x_location + i <= len(camera_buffer)-1:
                self.perception_field[int((self.ball_x_location + i) * perception_sample_adjust)] = 1
                self.perception_field[int((self.ball_x_location + i) * perception_sample_adjust)] = 1
                # camera_buffer[self.ball_x_location + i][int(y_center - self.ball_y_location)][0] = 0
                # camera_buffer[self.ball_x_location + i][int(y_center - self.ball_y_location)][1] = 0
                # camera_buffer[self.ball_x_location + i][int(y_center - self.ball_y_location)][2] = 0

        for i in range(self.perception_field_len):
            self.perception_field_surface[0][i][0] = int((self.perception_field[i]*0.5 + 0.5)*255)
            self.perception_field_surface[0][i][1] = int((self.perception_field[i]*0.5 + 0.5)*255)
            self.perception_field_surface[0][i][2] = int((self.perception_field[i]*0.5 + 0.5)*255)
            self.perception_field_surface[1][i][0] = int((self.perception_field[i]*0.5 + 0.5)*255)
            self.perception_field_surface[1][i][1] = int((self.perception_field[i]*0.5 + 0.5)*255)
            self.perception_field_surface[1][i][2] = int((self.perception_field[i]*0.5 + 0.5)*255)

        # print(self.perception_field)
        return camera_buffer

    def determine_reward(self):
        return self.reward

    def episode_finished(self):
        if self.collision or self.points > 100 or self.ticks > 3000:
            if self.points > self.record:
                self.record = self.points
            self.ticks = 0
            return 1.0
        else:
            return 0.0

    def reset(self):
        self.reward = 0
        self.points = 0
        self.collision = False
        self.phuc_translation.setSFVec3f(self.phuc_translation_home)
        self.phuc_rotation.setSFRotation(self.phuc_rotation_home)
        self.phuc_robot.resetPhysics()

        self.ping_pong_ball_translation.setSFVec3f(self.ping_pong_ball_home)
        self.ping_pong_ball.resetPhysics()

class PHUC_line_follower_wbt:
    def __init__(self, robot, camera_width, camera_height):
        self.robot = robot
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.collision = False

    def update(self):
        pass

    def determine_reward(self):
        pass

    def reset(self):
        pass

class PHUC_tu_wbt:
    def __init__(self, robot, camera_width, camera_height):
        self.robot = robot
        self.camera_width = camera_width
        self.camera_height = camera_height

        # Webots set wheel params
        self.rpm = 100
        self.rps = self.rpm / 60
        self.rad_per_sec = self.rps * 2 * 3.14159
        self.max_speed = self.rad_per_sec

        # reward scale of -1 to 1
        self.reward = [0.0,0.0]
        self.points = 0
        self.record = 0
        self.collision = False
        self.collision_magnitude = 0.0

        self.arena = self.robot.getFromDef('t_underground')
        self.arena_floor = self.arena.getField('floorSize')
        self.arena_size = self.arena_floor.getSFVec3f()
        self.arena_wall = self.arena.getField('wallHeight')
        self.arena_wall_height = self.arena_wall.getSFVec3f()
        # print(self.arena_wall_height)

        self.phuc_robot = self.robot.getFromDef('PHUC_robot')
        self.phuc_translation = self.phuc_robot.getField('translation')
        self.phuc_translation_home = self.phuc_translation.getSFVec3f()
        #used in update
        self.phuc_location = self.phuc_translation.getSFVec3f()
        self.phuc_rotation = self.phuc_robot.getField('rotation')
        self.phuc_rotation_home = self.phuc_rotation.getSFVec3f()
        self.left_wheel = self.robot.getDevice('wb_left_wheel')
        self.right_wheel = self.robot.getDevice('wb_right_wheel')
        self.phuc_inertial_unit = InertialUnit('PHUC_inertial_unit')
        self.phuc_inertial_unit.enable(100)
        self.phuc_orientation_pitch = self.phuc_inertial_unit.getRollPitchYaw()[1]
        self.phuc_orientation_yaw = self.phuc_inertial_unit.getRollPitchYaw()[2]
        # print(self.phuc_rotation_home)

        self.phuc_camera = self.robot.getDevice('wb_camera')
        self.camera_fov = self.phuc_camera.getFov()
        # divide fov by 8 individual camera segments representing radians per segment
        self.camera_pixels_per_radian = camera_width/self.camera_fov

        # FOV split into sqares radians per sqare
        self.camera_fov_segment = self.camera_fov/8
        self.camera_pixels_per_segment = self.camera_pixels_per_radian*self.camera_fov_segment
        # print(self.camera_fov)

        # Each segment 20x20 pixels of camera array 8x6 indicated as one hot encoded
        # ball = [1,0,0,0]
        # wall = [0,1,0,0]
        # floor = [0,0,1,0]
        # distance = [0,0,0,1]
        self.camera_flag_key = ['ball','wall','floor','distance']
        # initialize each segment as 'distance
        self.camera_flag_array = [[4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4],
                                  [4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4],[4,4,4,4,4,4,4,4]]

        self.ping_pong_ball = self.robot.getFromDef('ping_pong_ball_1')
        self.ping_pong_ball_translation = self.ping_pong_ball.getField('translation')
        self.ping_pong_ball_home = self.ping_pong_ball_translation.getSFVec3f()
        self.ping_pong_ball_appearance = self.robot.getFromDef('ppb_appearance')
        self.ping_pong_ball_color = self.ping_pong_ball_appearance.getField('baseColor')
        #print(self.ping_pong_ball_color.getSFVec3f())
        self.ping_pong_ball_colors = [[255/255,255/255,255/255],[14/255,237/255,152/255],[235/255,194/255,12/255],
                                      [46/255,28/255,186/255],[108/255,160/255,224/255],[242/255,167/255,208/255],
                                      [240/255,50/255,154/255],[43/255,153/255,127/255],[245/255, 59/255, 17/255],
                                      [245/255,241/255,10/255]]
        # self.current_color = self.ping_pong_ball_colors[random.randint(0,9)]
        # self.ping_pong_ball_color.setSFVec3f(self.current_color)
        self.current_color = [0.0, 1.0, 0.2]
        self.ping_pong_ball_color.setSFVec3f(self.current_color)

        #distance from ball x direction
        self.adjacent = 0.0
        #distance from ball y direction
        self.opposite = 0.0
        #angle difference between the PHUC camera orientation and the target ball location
        self.angle_difference = 0.0

        # integer values to be placed on camera buffer for display
        self.ball_x_location = 0
        self.ball_y_location = 0

        # creates an array size 100 representing the simplified visual field "state" of the robot
        # -1 maps to 0 representing wall 0 maps to 124 representing floor and 1 maps to 255 representing the target ball
        self.perception_field_len = 100
        self.perception_field = np.zeros(int(self.perception_field_len), dtype=np.float64)
        #pygame surface
        self.perception_field_surface = np.zeros((2,int(self.perception_field_len),3),dtype=np.uint8)

        self.ball_west = 0.0
        self.ball_east = 0.0
        self.ball_north = 0.0
        self.ball_south = 0.0

        self.ticks = 0
    # This is where the supervisor determines the location of all the objects
    # Reward/Punishment functions happen here
    # Relocate objects or reset if necessary
    def update(self):
        self.ticks += 1
        self.collision_magnitude = 0.0

        # first reward is for lw_vel
        # second reward is for rw_vel
        self.reward = [0.0, 0.0]

        # self.ball_west = 0.0
        # self.ball_east = 0.0
        # self.ball_north = 0.0
        # self.ball_south = 0.0

        lw_velocity = self.left_wheel.getVelocity()/self.max_speed
        rw_velocity = self.right_wheel.getVelocity() / self.max_speed
        if lw_velocity > 0.55:
            self.reward[0] += 0.5
        elif abs(lw_velocity) < 0.15:
            self.reward[0] -= 1.0
        elif lw_velocity < -0.5:
            self.reward[0] -= 0.5
        if rw_velocity > 0.55:
            self.reward[1] += 0.5
        elif abs(rw_velocity) < 0.15:
            self.reward[1] -= 1.0
        elif rw_velocity < -0.5:
            self.reward[1] -= 0.5

        self.phuc_location = self.phuc_translation.getSFVec3f()
        self.phuc_orientation_pitch = self.phuc_inertial_unit.getRollPitchYaw()[1]
        self.phuc_orientation_yaw = self.phuc_inertial_unit.getRollPitchYaw()[2]

        ppb_location = self.ping_pong_ball_translation.getSFVec3f()

        # last component corrects for camera offset
        self.adjacent = ppb_location[0] - self.phuc_location[0] - np.cos(self.phuc_orientation_yaw)*0.058
        self.opposite = ppb_location[1] - self.phuc_location[1] - np.sin(self.phuc_orientation_yaw)*0.058
        distance = math.sqrt(self.adjacent*self.adjacent + self.opposite*self.opposite)
        angle = np.arctan(self.opposite/(self.adjacent + 0.0001))
        #second quadrant
        if self.adjacent < 0 < self.opposite:
            angle = 3.1415 + angle
        #third quadrant
        elif self.adjacent < 0 and self.opposite < 0:
            angle = -3.1415 + angle
        # print("adjacent: " + str(adjacent))
        # print("opposite: " + str(opposite))
        # print("ball location: " + str(ppb_location))
        # print("phuc location: " + str(phuc_location))
        # print("phuc orientation: " + str(phuc_orientation))

        # print(self.phuc_orientation_yaw)

        self.angle_difference = self.phuc_orientation_yaw - angle
        if abs(self.angle_difference) < 0.15 and lw_velocity > 0.15 and rw_velocity > 0.15:
            if abs(lw_velocity - rw_velocity) < 0.03:
                self.reward[0] += 4.0
                self.reward[1] += 4.0
            elif abs(lw_velocity - rw_velocity) < 0.08:
                self.reward[0] += 3.0
                self.reward[1] += 3.0
            elif abs(lw_velocity - rw_velocity) < 0.13:
                self.reward[0] += 2.0
                self.reward[1] += 2.0
            elif abs(lw_velocity - rw_velocity) < 0.18:
                self.reward[0] += 1.0
                self.reward[1] += 1.0
            elif abs(lw_velocity - rw_velocity) < 0.23:
                self.reward[0] += 0.5
                self.reward[1] += 0.5
            else:
                self.reward[0] += 0.25
                self.reward[1] += 0.25
            if distance < 1.0:
                self.reward[0] += 1.5
                self.reward[1] += 1.5
            elif distance < 3.0:
                self.reward[0] += 1.0
                self.reward[1] += 1.0
            elif distance < 5.0:
                self.reward[0] += 0.5
                self.reward[1] += 0.5
        elif abs(self.angle_difference) < 0.34 and lw_velocity > 0.15 and rw_velocity > 0.15:
            if abs(lw_velocity - rw_velocity) < 0.03:
                self.reward[0] += 4.0
                self.reward[1] += 4.0
            elif abs(lw_velocity - rw_velocity) < 0.08:
                self.reward[0] += 3.0
                self.reward[1] += 3.0
            elif abs(lw_velocity - rw_velocity) < 0.13:
                self.reward[0] += 2.0
                self.reward[1] += 2.0
            elif abs(lw_velocity - rw_velocity) < 0.18:
                self.reward[0] += 1.0
                self.reward[1] += 1.0
            elif abs(lw_velocity - rw_velocity) < 0.23:
                self.reward[0] += 0.5
                self.reward[1] += 0.5
            else:
                self.reward[0] += 0.25
                self.reward[1] += 0.25
            if distance < 1.0:
                self.reward[0] += 1.5
                self.reward[1] += 1.5
            elif distance < 3.0:
                self.reward[0] += 1.0
                self.reward[1] += 1.0
            elif distance < 5.0:
                self.reward[0] += 0.5
                self.reward[1] += 0.5
        elif abs(self.angle_difference) > 0.40:
            self.reward[0] -= 0.25
            self.reward[1] -= 0.25
        # print("angle diff: " + str(self.angle_difference))

        if abs(self.adjacent) < 0.085 and abs(self.opposite) < 0.085 and abs(self.angle_difference) < 0.40:
            x_rand = random.uniform(-self.arena_size[0] / 2 + 2, self.arena_size[0] / 2 - 0.03)
            y_rand = random.uniform(-self.arena_size[1] / 2 + 1, self.arena_size[1] / 2 - 0.03)
            self.ping_pong_ball_translation.setSFVec3f([x_rand, y_rand, 0.3])
            # self.current_color = self.ping_pong_ball_colors[random.randint(0, 9)]
            # self.ping_pong_ball_color.setSFVec3f(self.current_color)
            self.ping_pong_ball.resetPhysics()
            self.points += 1
            self.ticks = 0
            self.reward[0] += 15.0
            self.reward[1] += 15.0

        # Wall collision detection
        if self.phuc_location[0] + 0.09 > self.arena_size[0] / 2 or self.phuc_location[0] - 0.09 < -self.arena_size[0] / 2:
            self.collision = True
            # self.collision_magnitude = 0.7
            self.reward[0] -= 10.0
            self.reward[1] -= 10.0
        elif self.phuc_location[0] + 0.18 > self.arena_size[0] / 2 or self.phuc_location[0] - 0.18 < -self.arena_size[0] / 2:
            self.reward[0] -= 1.0
            self.reward[1] -= 1.0
        if self.phuc_location[1] + 0.09 > self.arena_size[1] / 2 or self.phuc_location[1] - 0.09 < -self.arena_size[1] / 2:
            self.collision = True
            # self.collision_magnitude = 0.7
            self.reward[0] -= 10.0
            self.reward[1] -= 10.0
        elif self.phuc_location[1] + 0.18 > self.arena_size[1] / 2 or self.phuc_location[1] - 0.18 < -self.arena_size[1] / 2:
            self.reward[0] -= 1.0
            self.reward[1] -= 1.0

        print("reward: " + str(self.reward))
        #if ppb_location[0] - phuc_location[0] < -0.1:
        #    self.ball_west = 1.0
        #elif ppb_location[0] - phuc_location[0] > 0.1:
        #    self.ball_east = 1.0
        #if ppb_location[1] - phuc_location[1] < -0.1:
        #    self.ball_north = 1.0
        #elif ppb_location[1] - phuc_location[1] > 0.1:
        #    self.ball_north = 1.0
        # print(self.phuc_inertial_unit.getRollPitchYaw())

    def update_perception_field(self, camera_buffer):
        x_center = int(len(camera_buffer)/2)
        y_center = int(len(camera_buffer[0])/2)
        perception_sample_adjust = self.perception_field_len/self.camera_width

        # Sections of camera array to be sent into Conv2D network
        for i in range(-4,4):
            for j in range(-3,3):
                # print(str(x_center) + " " + str(y_center))
                camera_buffer[x_center + i * 20][y_center + j * 20][0] = 255
                camera_buffer[x_center + i * 20][y_center + j * 20][1] = 0
                camera_buffer[x_center + i * 20][y_center + j * 20][2] = 0

                camera_buffer[int(x_center + i*self.camera_pixels_per_segment)][int(y_center + j*self.camera_pixels_per_segment)][0] = 0
                camera_buffer[int(x_center + i*self.camera_pixels_per_segment)][int(y_center + j*self.camera_pixels_per_segment)][1] = 0
                camera_buffer[int(x_center + i*self.camera_pixels_per_segment)][int(y_center + j*self.camera_pixels_per_segment)][2] = 255

        self.ball_x_location = int(x_center + self.angle_difference*self.camera_pixels_per_radian)
        distance_to_ball = math.sqrt(self.opposite*self.opposite + self.adjacent*self.adjacent)
        ball_x_span = int(np.sin(0.02 /distance_to_ball)*self.camera_pixels_per_radian)
        self.ball_y_location = int(
            (np.arctan(-0.0076/distance_to_ball) + self.phuc_orientation_pitch) * self.camera_pixels_per_radian)
        # self.ball_y_location = int((np.arctan(0.02/distance_to_ball)+self.phuc_orientation_pitch)*self.camera_pixels_per_radian)

        #populate array with wall/floor values
        #separate into facing top or bottom wall

        pi = 3.14159
        pi_over_2 = 3.14159/2
        pi_over_4 = 3.14159/4
        pi_3_over_4 = 3.14159*3/4
        distance_top = abs(self.arena_size[1] / 2 - self.phuc_location[1] - np.sin(self.phuc_orientation_yaw) * 0.058)
        distance_bottom = abs(-self.arena_size[1] / 2 - self.phuc_location[1] - np.sin(self.phuc_orientation_yaw) * 0.058)
        distance_left = abs(-self.arena_size[0] / 2 - self.phuc_location[0] - np.cos(self.phuc_orientation_yaw) * 0.058)
        distance_right = abs(self.arena_size[0] / 2 - self.phuc_location[0] - np.cos(self.phuc_orientation_yaw) * 0.058)
        ru_angle = np.arctan(distance_top/distance_right)
        # print("ru: " + str(ru_angle))
        rl_angle = -np.arctan(distance_bottom/distance_right)
        #print("rl: " + str(rl_angle))
        lu_angle = np.arctan(distance_left/distance_top) + pi_over_2
        #print("lu: " + str(lu_angle))
        ll_angle = -np.arctan(distance_left/distance_bottom) - pi_over_2
        #print("ll: " + str(ll_angle))

        # facing top wall
        if ru_angle <= self.phuc_orientation_yaw <= lu_angle:
            for i in range(-50, 50):
                angle = self.phuc_orientation_yaw - i/self.camera_pixels_per_radian
                if angle <= ru_angle:
                    distance_to_wall = distance_right / np.cos(angle)
                elif angle >= lu_angle:
                    angle = abs(angle - pi)
                    distance_to_wall = distance_left / np.cos(angle)
                else:
                    angle = abs(angle - pi_over_2)
                    distance_to_wall = distance_top/np.cos(angle)
                self.perception_field[50 + i] = -np.exp(-(distance_to_wall*1.8))
        # facing bottom wall
        elif ll_angle <= self.phuc_orientation_yaw <= rl_angle:
            for i in range(-50, 50):
                angle = self.phuc_orientation_yaw - i / self.camera_pixels_per_radian
                if angle <= ll_angle:
                    angle = abs(angle + pi)
                    distance_to_wall = distance_left / np.cos(angle)
                elif angle >= rl_angle:
                    angle = abs(angle)
                    distance_to_wall = distance_right / np.cos(angle)
                else:
                    angle = abs(angle + pi_over_2)
                    distance_to_wall = distance_bottom / np.cos(angle)
                self.perception_field[50 + i] = -np.exp(-(distance_to_wall*1.8))
        # facing right wall
        elif rl_angle <= self.phuc_orientation_yaw <= ru_angle:
            for i in range(-50, 50):
                angle = self.phuc_orientation_yaw - i / self.camera_pixels_per_radian
                if angle >= ru_angle:
                    angle = abs(angle - pi_over_2)
                    distance_to_wall = distance_top / np.cos(angle)
                elif angle <= rl_angle:
                    angle = abs(angle + pi_over_2)
                    distance_to_wall = distance_bottom / np.cos(angle)
                else:
                    angle = abs(angle)
                    distance_to_wall = distance_right / np.cos(angle)
                self.perception_field[50 + i] = -np.exp(-(distance_to_wall*1.8))
        # facing left wall
        else:
            if self.phuc_orientation_yaw > 0:
                lw_angle = pi - self.phuc_orientation_yaw
            else:
                lw_angle = -(pi + self.phuc_orientation_yaw)
            for i in range(-50, 50):
                angle = lw_angle + i / self.camera_pixels_per_radian
                if abs(ll_angle) - pi >= angle:
                    angle = abs(angle + pi_over_2)
                    distance_to_wall = distance_bottom / np.cos(angle)
                elif abs(ll_angle) - pi <= angle <= pi - lu_angle:
                    angle = abs(angle)
                    distance_to_wall = distance_left / np.cos(angle)
                else:
                    angle = abs(angle - pi_over_2)
                    distance_to_wall = distance_top / np.cos(angle)
                self.perception_field[50 + i] = -np.exp(-(distance_to_wall*1.8))

        #for i in range(self.perception_field_len):
        #    self.perception_field[i] = -0.5
        if 0 <= self.ball_x_location <= len(camera_buffer) - 1:
            camera_buffer[self.ball_x_location][int(y_center - self.ball_y_location)][0] = 0
            camera_buffer[self.ball_x_location][int(y_center - self.ball_y_location)][1] = 0
            camera_buffer[self.ball_x_location][int(y_center - self.ball_y_location)][2] = 0
        for i in range(-ball_x_span,ball_x_span+1):
            if 0 <= self.ball_x_location + i <= len(camera_buffer)-1:
                self.perception_field[int((self.ball_x_location + i) * perception_sample_adjust)] = 1
                self.perception_field[int((self.ball_x_location + i) * perception_sample_adjust)] = 1
                # camera_buffer[self.ball_x_location + i][int(y_center - self.ball_y_location)][0] = 0
                # camera_buffer[self.ball_x_location + i][int(y_center - self.ball_y_location)][1] = 0
                # camera_buffer[self.ball_x_location + i][int(y_center - self.ball_y_location)][2] = 0

        for i in range(self.perception_field_len):
            self.perception_field_surface[0][i][0] = int((self.perception_field[i]*0.5 + 0.5)*255)
            self.perception_field_surface[0][i][1] = int((self.perception_field[i]*0.5 + 0.5)*255)
            self.perception_field_surface[0][i][2] = int((self.perception_field[i]*0.5 + 0.5)*255)
            self.perception_field_surface[1][i][0] = int((self.perception_field[i]*0.5 + 0.5)*255)
            self.perception_field_surface[1][i][1] = int((self.perception_field[i]*0.5 + 0.5)*255)
            self.perception_field_surface[1][i][2] = int((self.perception_field[i]*0.5 + 0.5)*255)

        # print(self.perception_field)
        return camera_buffer

    def determine_reward(self):
        return self.reward

    def episode_finished(self):
        if self.collision or self.points > 100 or self.ticks > 3000:
            if self.points > self.record:
                self.record = self.points
            self.ticks = 0
            return 1.0
        else:
            return 0.0

    def reset(self):
        self.reward = 0
        self.points = 0
        self.collision = False
        self.phuc_translation.setSFVec3f(self.phuc_translation_home)
        self.phuc_rotation.setSFRotation(self.phuc_rotation_home)
        self.phuc_robot.resetPhysics()

        self.ping_pong_ball_translation.setSFVec3f(self.ping_pong_ball_home)
        self.ping_pong_ball.resetPhysics()

class PHUC_real_world:
    def __init__(self):
        self.reward = 0
        self.points = 0
        self.record = 0
        self.collision = False

    def update(self):
        pass

    def determine_reward(self):
        return self.reward

    def reset(self):
        pass