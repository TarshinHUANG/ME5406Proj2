# =============================================================================
#   Filename         : Ball_env.py
#
#   Created On       : 2021-10-16 1:33
#
#   Last Modified    :
#
#   Revision         : 1.0
#
#   Author           : HuangYeshun
#
#   Description      : The env based on velocity
# =============================================================================
# import
import gym
import pyglet
from pyglet import image
import math
import random
from pyglet.libs.win32.constants import DISP_CHANGE_BADDUALVIEW, TRUE
from sympy import symbols, Eq, solve
import datetime

# define some buttons
RAM_BALL = True  # True-random location of the football

# define some parameters
MAP_WIDTH = 800  # the width of the map
MAP_LENGTH = 800  # the length of the map
ROB_POS = (300, 300, 0)  # the initial location of robot x, y, θ (valid if random location is closed)
BALL_POS = (500, 500)  # the initial location of football (valid if random location is closed)
GATE_POS = (700,700)  # the initial location of gate (valid if random location is closed), this can be n-D array for n gates
DT = 0.01  # timespan for each round
MAX_STEPS = 100000  # the max_steps for each episode

# Adjust
ROB_SIZE = 50   # the radius of robot (supposing circle)
BALL_SIZE = 50  # the radius of football (supposing circle)
CE_FRI = 0.01   # coefficient of friction
                # range[0,1]. 0-no friction
MASS_RAT = 1    # the ratio of robot to football
                # range(0,inf] inf-robot is far heavier than football
SLD_THD = 0     # the random sliding's threshold when robot knicking the ball
                # range[0,inf] 0-there is no random sliding on robot
VEL_THRD = 100  # the max velocity is considered as entering the gate
SPEED_THRD = 500  # the max speed of the robot and the football
ANG_SPEED_THRD = 20  # the max angular speed of the robot
 
# Reward adjustment
REACH_GATE = 1000  # when the ball reaches the gate
REACH_BALL = 100   # when the robot hits the ball
HIT_WALL = 0  # the robot hits the wall
DIS_RB_N = 1  # the nearer distance reward between the robot and the ball
DIS_RB_F = -1  # the farther distance reward between the robot and the ball
STEP = -3  # each step
'''
    Oberservation:
        robot position, football position, robot speed, football speed, robot orientation, gate position
    Action:
        robot speed
'''


class Ball_env(gym.Env):
    # inherit three properties from gym - action_space, observation_space, reward_range
    # inherit 5 methods from gym - step, reset, render, close, seed

    viewer = None  # initialize viewer as none
    old_distanceBG = -1  # store the distance between the football and the gate
    old_distanceRB = -1  # store the distance between the robot and the football

    def __init__(self):
        self.gate_pos = [0, 0]
        self.ball_pos = [0, 0]
        self.rob_pos = [0, 0, 0]
        self.gate_pos[0] = GATE_POS[0]  # gate position
        self.gate_pos[1] = GATE_POS[1]  # gate position
        if RAM_BALL:
            self.ball_pos[0] = round(random.uniform(0,MAP_LENGTH))
            self.ball_pos[1] = round(random.uniform(0,MAP_WIDTH))
        else:
            self.ball_pos[0] = BALL_POS[0]  # ball position
            self.ball_pos[1] = BALL_POS[1]  # ball position
        self.ball_vel = [0, 0]  # ball velocity, vx, vy
        self.rob_pos[0] = ROB_POS[0]  # robot position
        self.rob_pos[1] = ROB_POS[1]  # robot position
        self.rob_pos[2] = ROB_POS[2]  # robot position
        self.rob_vel = [0, 0, 0]  # robot velocity, vx, vy, ω
        self.rob_wheel_vel = [0, 0]  # robot wheel velocity
        self.steps = 0  # the steps
        self.ignore_hitting = False  # avoid bug when hitting
        self.reward = 0  # the reward
        self.col_type = 0  # collision type
        self.label = pyglet.text.Label("reward:",
                                       font_name='Times New Roman',
                                       font_size=18,
                                       x=18 * 4, y=MAP_WIDTH - 36,
                                       anchor_x='center', anchor_y='center')

    # reset environment
    def reset(self):
        self.gate_pos[0] = GATE_POS[0]  # gate position
        self.gate_pos[1] = GATE_POS[1]  # gate position
        self.ball_vel = [0, 0]  # ball velocity, vx, vy
        if RAM_BALL:
            self.ball_pos[0] = round(random.uniform(BALL_SIZE,MAP_LENGTH-BALL_SIZE))
            self.ball_pos[1] = round(random.uniform(BALL_SIZE,MAP_WIDTH-BALL_SIZE))
        else:
            self.ball_pos[0] = BALL_POS[0]  # ball position
            self.ball_pos[1] = BALL_POS[1]  # ball position
        self.rob_pos[0] = ROB_POS[0]  # robot position
        self.rob_pos[1] = ROB_POS[1]  # robot position
        self.rob_pos[2] = ROB_POS[2]  # robot position
        self.rob_vel = [0, 0, 0]  # robot velocity, vx, vy, ω
        self.rob_wheel_vel = [0, 0]  # robot wheel velocity
        self.steps = 0  # the steps
        self.ignore_hitting = False  # avoid bug when hitting
        self.reward = 0  # the reward
        self.col_type = 0  # collision type
        return [self.gate_pos[0], self.gate_pos[1], self.ball_pos[0], self.ball_pos[0], self.ball_vel[0],
                self.ball_vel[1], self.rob_pos[0], self.rob_pos[1], self.rob_pos[2],
                self.rob_vel[0], self.rob_vel[1], self.rob_vel[2]]

    @staticmethod
    def action_two_wheel(action_class):
        action = [0, 0]
        if action_class == 0 or action_class == 1 or action_class == 2:
            action[0] = -10
        elif action_class == 3 or action_class == 4 or action_class == 5:
            action[0] = 0
        elif action_class == 6 or action_class == 7 or action_class == 8:
            action[0] = 10
        if action_class == 0 or action_class == 3 or action_class == 6:
            action[1] = -10
        elif action_class == 1 or action_class == 4 or action_class == 7:
            action[1] = 0
        elif action_class == 2 or action_class == 5 or action_class == 8:
            action[1] = 10
        else:
            print('Wrong action input!!!')
        return action

    def step(self, action_class):
        action = self.action_two_wheel(action_class)
        self.update_velocity(action)  # update the velocity for football and robot
        self.update_position()  # update the position for football and robot
        self.reward = self.get_reward()  # calculate the rewards for each state and action
        done = self.is_end()

        self.steps = self.steps + 1  # update steps
        return [self.gate_pos[0], self.gate_pos[1], self.ball_pos[0], self.ball_pos[0], self.ball_vel[0],
                self.ball_vel[1], self.rob_pos[0], self.rob_pos[1], self.rob_pos[2],
                self.rob_vel[0], self.rob_vel[1], self.rob_vel[2]], self.reward, done

    # visualize
    def render(self):
        if self.viewer is None:  # if there is no viewer, then create one
            self.viewer = Viewer(self.rob_pos, self.ball_pos, self.gate_pos)
        self.viewer.render(self.rob_pos, self.ball_pos, self.gate_pos, self.reward)

    def update_velocity(self, action):
        # update wheel velocity based on action
        self.rob_wheel_vel[0] = self.rob_wheel_vel[0] + action[0]
        self.rob_wheel_vel[1] = self.rob_wheel_vel[1] + action[1]

        # transfer wheel velocity to robot velocity
        rob_vel_liear = 0.5 * (self.rob_wheel_vel[0] + self.rob_wheel_vel[1])  # get the linear velocity
        self.rob_vel[2] = 0.5 * (self.rob_wheel_vel[0] - self.rob_wheel_vel[1]) / ROB_SIZE  # get the angular velocity
        self.rob_vel[0] = rob_vel_liear * math.sin(self.rob_pos[2])  # get vx
        self.rob_vel[1] = rob_vel_liear * math.cos(self.rob_pos[2])  # get vy

        self.col_type = self.collision_detect()
        # changing velocity according to collision
        if self.col_type == 0:
            # 0-no collision
            pass
        elif self.col_type == 1:
            # 1-collision between robot and football
            n = [self.ball_pos[0] - self.rob_pos[0],
                 self.ball_pos[1] - self.rob_pos[1]]  # The n vector, points from robot to football
            x = [1, 0]  # x axis orientation
            phi = math.asin(
                (x[0] * n[1] - x[1] * n[0]) / math.sqrt(n[1] * n[1] + n[0] * n[0]))  # the angle between n and x
            if n[0] < 0:  # phi is in the 2 and 3 quadrants
                phi = math.pi - phi
            else:  # phi is in the 1 and 4 quadrants
                pass
            # robot velocity in n coordinate
            rob_vel_n = [math.cos(phi) * self.rob_vel[0] + math.sin(phi) * self.rob_vel[1],
                         -math.sin(phi) * self.rob_vel[0] + math.cos(phi) * self.rob_vel[1]]
            # football velocity in n coordinate
            ball_vel_n = [math.cos(phi) * self.ball_vel[0] + math.sin(phi) * self.ball_vel[1],
                          -math.sin(phi) * self.ball_vel[0] + math.cos(phi) * self.ball_vel[1]]

            ## calculation the new velocity after collision
            # define new velocity after collision
            rob_vel_n_new = [0, 0]
            ball_vel_n_new = [0, 0]
            # conservation of momentum
            rob_vel_n_new[1] = rob_vel_n[1]
            ball_vel_n_new[1] = ball_vel_n[1]

            # # method1
            # start_time=datetime.datetime.now()
            # # x: rob_vel_n_new[0], y: ball_vel_n_new[0]
            # x,y = symbols('x y')
            # # conservation of momentum
            # eq1 = Eq(MASS_RAT*x+y-MASS_RAT*rob_vel_n[0]-ball_vel_n[0])
            # # conservation of mechanical energy
            # eq2 = Eq(MASS_RAT*  (x*x)+(y*y)-
            #         MASS_RAT*   (rob_vel_n[0]*rob_vel_n[0])-
            #                     (ball_vel_n[0]*ball_vel_n[0]))
            # sol=solve((eq1,eq2),(x,y))
            # end_time=datetime.datetime.now()
            # print(sol,end_time-start_time)

            # method2
            start_time = datetime.datetime.now()
            M = MASS_RAT
            a = MASS_RAT * rob_vel_n[0] + ball_vel_n[0]
            b = MASS_RAT * (rob_vel_n[0] * rob_vel_n[0]) + (ball_vel_n[0] * ball_vel_n[0])
            x1 = (M * a - math.sqrt(b * M + b * M * M - a * a * M)) / (M + M * M)
            x2 = (M * a + math.sqrt(b * M + b * M * M - a * a * M)) / (M + M * M)
            y1 = a - M * x1
            y2 = a - M * x2
            end_time = datetime.datetime.now()
            # print(x1, y1, x2, y2, end_time - start_time)

            # choose the solution
            if abs(x1 - rob_vel_n[0]) + abs(y1 - ball_vel_n[0]) <= 0.001:
                # unwanted solution
                rob_vel_n_new[0] = x2
                ball_vel_n_new[0] = y2
            else:
                # wanted solution
                rob_vel_n_new[0] = x1
                ball_vel_n_new[0] = y1

            ## get the velocity in x,y coordinate through rotation matrix
            # robot velocity in x,y coordinate
            self.rob_vel = [math.cos(phi) * rob_vel_n_new[0] - math.sin(phi) * rob_vel_n_new[1],
                            math.sin(phi) * rob_vel_n_new[0] + math.cos(phi) * rob_vel_n_new[1], self.rob_vel[2]]
            # football velocity in x,y coordinate
            self.ball_vel = [math.cos(phi) * ball_vel_n_new[0] - math.sin(phi) * ball_vel_n_new[1],
                             math.sin(phi) * ball_vel_n_new[0] + math.cos(phi) * ball_vel_n_new[1]]


        elif self.col_type == 2:
            # 2-collision between robot and wall
            if self.rob_pos[0] <= 0 + ROB_SIZE or self.rob_pos[0] >= MAP_LENGTH - ROB_SIZE:
                # hit in the x axis
                self.rob_vel[0] = -self.rob_vel[0]
            if self.rob_pos[1] <= 0 + ROB_SIZE or self.rob_pos[1] >= MAP_WIDTH - ROB_SIZE:
                # hit in the y axis
                self.rob_vel[1] = -self.rob_vel[1]

        elif self.col_type == 3:
            # 3-collision between football and wall
            if self.ball_pos[0] <= 0 + BALL_SIZE or self.ball_pos[0] >= MAP_LENGTH - BALL_SIZE:
                # hit in the x axis
                self.ball_vel[0] = -self.ball_vel[0]
            if self.ball_pos[1] <= 0 + BALL_SIZE or self.ball_pos[1] >= MAP_WIDTH - BALL_SIZE:
                # hit in the y axis
                self.ball_vel[1] = -self.ball_vel[1]

        else:
            pass

        # changing velocity according to friction
        # changing robot velocity
        self.rob_vel[0] = self.rob_vel[0] - CE_FRI * self.rob_vel[0]
        self.rob_vel[1] = self.rob_vel[1] - CE_FRI * self.rob_vel[1]
        self.rob_vel[2] = self.rob_vel[2] - CE_FRI * self.rob_vel[2]
        # changing football velocity
        self.ball_vel[0] = self.ball_vel[0] - CE_FRI * self.ball_vel[0]
        self.ball_vel[1] = self.ball_vel[1] - CE_FRI * self.ball_vel[1]

        # changing velocity according to sliding when the robot hits the football
        if self.col_type == 1:  # the robot hits the football
            self.ball_vel[0] = self.ball_vel[0] + random.uniform(-1, 1) * SLD_THD
            self.ball_vel[1] = self.ball_vel[1] + random.uniform(-1, 1) * SLD_THD

        global speed_penalty
        speed_penalty = False  # over speed penalty of reward function
        # limit the velocity
        if self.rob_vel[0] >= SPEED_THRD:
            self.rob_vel[0] = SPEED_THRD
            speed_penalty = True
        else:
            pass
        if self.rob_vel[0] <= -SPEED_THRD:
            self.rob_vel[0] = -SPEED_THRD
            speed_penalty = True
        else:
            pass
        if self.rob_vel[1] >= SPEED_THRD:
            self.rob_vel[1] = SPEED_THRD
            speed_penalty = True
        else:
            pass
        if self.rob_vel[1] <= -SPEED_THRD:
            self.rob_vel[1] = -SPEED_THRD
            speed_penalty = True
        else:
            pass
        if self.rob_vel[2] >= ANG_SPEED_THRD:
            self.rob_vel[2] = ANG_SPEED_THRD
            speed_penalty = True
        else:
            pass
        if self.ball_vel[0] >= SPEED_THRD:
            self.ball_vel[0] = SPEED_THRD
            speed_penalty = True
        else:
            pass
        if self.ball_vel[0] <= -SPEED_THRD:
            self.ball_vel[0] = -SPEED_THRD
            speed_penalty = True
        else:
            pass
        if self.ball_vel[1] >= SPEED_THRD:
            self.ball_vel[1] = SPEED_THRD
            speed_penalty = True
        else:
            pass
        if self.ball_vel[1] <= -SPEED_THRD:
            self.ball_vel[1] = -SPEED_THRD
            speed_penalty = True
        else:
            pass

        # transfer robot velocity to wheel velocity
        rob_vel_liear = math.sqrt(self.rob_vel[0] * self.rob_vel[0] + self.rob_vel[1] * self.rob_vel[1])

        if self.rob_vel[1] == 0:
            # in case atan will have no solutions
            if self.rob_vel[0] >= 0:
                self.rob_pos[2] = 0
            else:
                self.rob_pos[2] = math.pi
        else:
            theta = math.atan(self.rob_vel[0] / self.rob_vel[1])
            if self.rob_vel[1] > 0:
                self.rob_pos[2] = theta
            else:
                self.rob_pos[2] = math.pi + theta
        self.rob_wheel_vel[0] = rob_vel_liear + self.rob_vel[2] * ROB_SIZE
        self.rob_wheel_vel[1] = rob_vel_liear - self.rob_vel[2] * ROB_SIZE

        # get the v1, v2 from vx, vy

    def update_position(self):
        self.ignore_hitting = False
        if self.distance(self.rob_pos, self.ball_pos) <= (ROB_SIZE + BALL_SIZE):
            temp_rob_pos = [0, 0, 0]
            temp_ball_pos = [0, 0]
            temp_rob_pos[0] = self.rob_pos[0] + self.rob_vel[0] * DT
            temp_rob_pos[1] = self.rob_pos[1] + self.rob_vel[1] * DT
            temp_rob_pos[2] = self.rob_pos[2] + self.rob_vel[2] * DT
            temp_ball_pos[0] = self.ball_pos[0] + self.ball_vel[0] * DT
            temp_ball_pos[1] = self.ball_pos[1] + self.ball_vel[1] * DT

            if self.distance(temp_rob_pos, temp_ball_pos) <= (ROB_SIZE + BALL_SIZE):
                self.ignore_hitting = True
            else:
                self.ignore_hitting = False

        else:
            self.ignore_hitting = False

        # the position of robot
        self.rob_pos[0] = self.rob_pos[0] + self.rob_vel[0] * DT
        self.rob_pos[1] = self.rob_pos[1] + self.rob_vel[1] * DT
        self.rob_pos[2] = self.rob_pos[2] + self.rob_vel[2] * DT
        # the position of football
        self.ball_pos[0] = self.ball_pos[0] + self.ball_vel[0] * DT
        self.ball_pos[1] = self.ball_pos[1] + self.ball_vel[1] * DT

        # limit the value of rob_pos and ball_pos to aviod stucking into the wall
        if self.rob_pos[0] <= 0 + ROB_SIZE:
            self.rob_pos[0] = ROB_SIZE
        if self.rob_pos[0] >= MAP_LENGTH - ROB_SIZE:
            self.rob_pos[0] = MAP_LENGTH - ROB_SIZE
        if self.rob_pos[1] <= 0 + ROB_SIZE:
            self.rob_pos[1] = ROB_SIZE
        if self.rob_pos[1] >= MAP_WIDTH - ROB_SIZE:
            self.rob_pos[1] = MAP_WIDTH - ROB_SIZE

        if self.ball_pos[0] <= 0 + BALL_SIZE:
            self.ball_pos[0] = BALL_SIZE
        if self.ball_pos[0] >= MAP_LENGTH - BALL_SIZE:
            self.ball_pos[0] = MAP_LENGTH - BALL_SIZE
        if self.ball_pos[1] <= 0 + BALL_SIZE:
            self.ball_pos[1] = BALL_SIZE
        if self.ball_pos[1] >= MAP_WIDTH - BALL_SIZE:
            self.ball_pos[1] = MAP_WIDTH - BALL_SIZE

    def get_reward(self):
        # get the new distance between the ball and the gate
        new_distanceBG = self.distance(self.ball_pos, self.gate_pos)
        new_distanceRB = self.distance(self.rob_pos, self.ball_pos)
        if self.old_distanceBG == -1:
            # first value of old_distance
            self.old_distanceBG = new_distanceBG
        if self.old_distanceRB == -1:
            # first value of old_distance
            self.old_distanceRB = new_distanceRB
        if (self.distance(self.ball_pos, self.gate_pos) <= BALL_SIZE and self.ball_vel[0] <= VEL_THRD and
                self.ball_vel[1] <= VEL_THRD):
            # the football reach the gate
            reward = REACH_GATE
        elif self.col_type == 1:
            # hits the ball
            reward = REACH_BALL
        elif self.col_type == 2:
            # the robot hits the wall
            reward = HIT_WALL
        elif (new_distanceRB < self.old_distanceRB):
            # get closer to the ball
            reward = DIS_RB_N
        elif (new_distanceRB > self.old_distanceRB):
            # get farther to the ball
            reward = DIS_RB_F
        else:
            # every step has penalty
            reward = STEP
        # store the new distance between the football and the gate
        self.old_distanceBG = new_distanceBG
        # store the new distance between the robot and the football
        self.old_distanceRB = new_distanceRB

        return reward

    # generate random action for testing
    def random_action(self):
        action = random.randint(0, 8)
        return action

    # detect collision
    def collision_detect(self):
        col_type = 0  # 0-no collision,
        # 1-collision between robot and football
        # 2-collision between robot and wall
        # 3-collision between football and wall
        if self.distance(self.rob_pos, self.ball_pos) <= (ROB_SIZE + BALL_SIZE) and self.ignore_hitting == False:
            col_type = 1
        if self.rob_pos[0] <= 0 + ROB_SIZE or self.rob_pos[0] >= MAP_LENGTH - ROB_SIZE or self.rob_pos[
            1] <= 0 + ROB_SIZE or self.rob_pos[1] >= MAP_WIDTH - ROB_SIZE:
            col_type = 2
        if self.ball_pos[0] <= 0 + BALL_SIZE or self.ball_pos[0] >= MAP_LENGTH - BALL_SIZE or self.ball_pos[
            1] <= 0 + BALL_SIZE or self.ball_pos[1] >= MAP_WIDTH - BALL_SIZE:
            col_type = 3

        return col_type

    # detect episode end
    def is_end(self):
        is_end = False
        # situation1: the robot hits the football
        if (self.distance(self.ball_pos, self.gate_pos) <= BALL_SIZE and self.ball_vel[0] <= VEL_THRD and
                self.ball_vel[1] <= VEL_THRD):
            is_end = True
        # # situation2: too much steps
        # if (self.steps > MAX_STEPS):
        #     is_end = True
        return is_end

    # get distance between two points
    def distance(self, posA, posB):
        return math.sqrt(math.pow(posA[0] - posB[0], 2) + math.pow(posA[1] - posB[1], 2))


class Viewer(pyglet.window.Window):
    frame_cnt = 0  # count the frame

    def __init__(self, rob_pos, ball_pos, gate_pos):
        # vsync=False to not use the monitor FPS (75Hz) as operating rate, instead, the cpu will determine the operating rate
        super(Viewer, self).__init__(width=MAP_LENGTH, height=MAP_WIDTH, resizable=True, caption='FootballCourt',
                                     vsync=False)

        pyglet.gl.glClearColor(0.5, 0.5, 0.5, 0.5)  # the windows color
        # self.background_img = background_image

        self.batch = pyglet.graphics.Batch()  # display whole batch at once

        # load football
        # load the picture, cannot use pyglet.image.load()
        self.ball = pyglet.resource.image('soccer.png')
        self.ball.width = 2 * BALL_SIZE
        self.ball.height = 2 * BALL_SIZE
        self.ball.anchor_x = self.ball.width // 2
        self.ball.anchor_y = self.ball.height // 2
        # load robot
        # load the picture, cannot use pyglet.image.load()
        self.rob = pyglet.resource.image('robot.png')
        self.rob.width = 2 * ROB_SIZE
        self.rob.height = 2 * ROB_SIZE
        self.rob.anchor_x = self.rob.width // 2
        self.rob.anchor_y = self.rob.height // 2
        # load gate
        # load the picture, cannot use pyglet.image.load()
        self.gate = pyglet.resource.image('gate.png')
        self.gate.width = 50
        self.gate.height = 50
        self.gate.anchor_x = self.gate.width // 2
        self.gate.anchor_y = self.gate.height // 2

    # refresh and show the screen
    def render(self, rob_pos, ball_pos, gate_pos, reward):
        # arrange the handlers, otherwise the ui will stuck
        self.dispatch_events()
        # store pictures in the batch
        senario = []
        # robot position
        rob_batch = pyglet.sprite.Sprite(img=self.rob, x=rob_pos[0], y=rob_pos[1], batch=self.batch)
        # robot rotation
        rob_batch.rotation = rob_pos[2] * 180 / math.pi
        # add robot to the batch
        senario.append(rob_batch)
        # ball position
        ball_batch = pyglet.sprite.Sprite(img=self.ball, x=ball_pos[0], y=ball_pos[1], batch=self.batch)
        # add football to the batch
        senario.append(ball_batch)
        # gate position
        gate_batch = pyglet.sprite.Sprite(img=self.gate, x=gate_pos[0], y=gate_pos[1], batch=self.batch)
        # add gate to the batch
        senario.append(gate_batch)

        # clean the screen
        self.clear()
        # draw the scenarios
        self.batch.draw()

        # show the rewards
        if abs(reward) != 0.01:
            self.label = pyglet.text.Label("reward:" + str(reward),
                                           font_name='Times New Roman',
                                           font_size=18,
                                           x=18 * 4, y=MAP_WIDTH - 36,
                                           anchor_x='center', anchor_y='center')
        else:
            self.frame_cnt = self.frame_cnt + 1

        if self.frame_cnt >= 500:
            self.frame_cnt = 0
            self.label = pyglet.text.Label("reward:",
                                           font_name='Times New Roman',
                                           font_size=18,
                                           x=18 * 4, y=MAP_WIDTH - 36,
                                           anchor_x='center', anchor_y='center')

        self.label.draw()

        # swap front and back buffers to unpdate the visible display with the back buffer
        self.flip()


if __name__ == '__main__':
    env = Ball_env()

    while True:

        for i in range(100):
            env.render()
            env.step(env.random_action())
        env.reset()
        
