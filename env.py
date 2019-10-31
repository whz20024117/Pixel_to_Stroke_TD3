import gym
from gym import spaces
import numpy as np
from config import config
import tensorflow as tf
from pretrianGAN.utils import discriminator, CNN
from skimage.draw import line, bezier_curve
from matplotlib import pyplot as plt


class SketchDiscriminator:
    def __init__(self, path):
        self.X = tf.placeholder(tf.float32, shape=[None] + config['STATE_DIM'] + [1], name='X')
        self.score = discriminator(self.X, rate=0.0)

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(path))

    def inference(self, X):
        X = X.reshape([-1] + config['STATE_DIM'] + [1])
        result = self.sess.run(self.score, feed_dict={self.X: X})
        return X.reshape(config['STATE_DIM']), result

    def get_score(self, X):
        X = X.reshape([-1] + config['STATE_DIM'] + [1])
        _, scores = self.inference(X)
        score = scores[0][0]

        return score


class SketchClassifier:
    def __init__(self, path):
        self.X = tf.placeholder(tf.float32, shape=[None] + config['STATE_DIM'] + [1], name='X')
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.score = CNN(self.X, self.keep_prob)

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(path))

    def inference(self, X):
        X = X.reshape([-1] + config['STATE_DIM'] + [1])
        result = self.sess.run(self.score, feed_dict={self.X: X, self.keep_prob: 1.0})
        return X.reshape(config['STATE_DIM']), result

    def get_score(self, X):
        X = X.reshape([-1] + config['STATE_DIM'] + [1])
        _, scores = self.inference(X)
        score = scores[0][0]

        return score


class SketchDesigner(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, classifier, max_T=config['MAX_T']):
        super(SketchDesigner, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(14,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(config['STATE_DIM'][0], config['STATE_DIM'][0], 1), dtype=np.uint8)
        self.classifier = classifier
        self.stroke_count = 0
        self.t = 0
        self.dim = self.observation_space.shape
        self.canvas = np.zeros(self.dim)
        self.max_T = max_T
        self.terminal = False
        self.previous_score = 0.25
        self.policy = None

    def get_policy(self, policy):
        assert self.policy is None
        self.policy = policy

    def draw(self, action):
        action = np.clip(action, -1, 1)
        a = scale(action[0:3], -1, 1, 0, 1)
        b = scale(action[3:13], -1, 1, 0, config['STATE_DIM'][0] - 1)
        c = scale(action[13:14], -1, 1, 0, 4)
        action = np.concatenate([a, b, c]).reshape(config['ACTION_DIM'],)

        # Parameter Validation and noises
        action_category = np.argmax(action[0:3])
        if self.stroke_count == 0:
            axis = np.asarray(action[3:13], dtype=np.uint8) + np.int_(np.random.normal(0, 2, action[3:13].shape[0]))
            c_p = action[13] + np.random.normal(0, 1)
        else:
            axis = np.asarray(action[3:13], dtype=np.uint8)
            c_p = action[13]

        for i in range(axis.shape[0]):
            if axis[i] < 2:
                axis[i] = 2
            elif axis[i] >= self.dim[0] - 2:
                axis[i] = self.dim[0] - 2

        if action_category == 1:
            self.stroke_count += 1
            # Draw line
            rr, cc = line(axis[0], axis[1], axis[2], axis[3])
            self.canvas[rr, cc] = 255

        if action_category == 2:
            self.stroke_count += 1
            # Draw Curve
            try:
                rr, cc = bezier_curve(axis[4], axis[5],
                                  axis[6], axis[7],
                                  axis[8], axis[9],
                                  c_p)
            except MemoryError:
                while True:
                    try:
                        _x1, _y1 = move_point(axis[4], axis[5])
                        _x2, _y2 = move_point(axis[6], axis[7])
                        _x3, _y3 = move_point(axis[8], axis[9])
                        rr, cc = bezier_curve(_x1, _y1,
                                              _x2, _y2,
                                              _x3, _y3,
                                              c_p)
                        break
                    except MemoryError:
                        continue

            try:
                self.canvas[rr, cc] = 255
            except IndexError:
                rr = np.clip(rr, 0, config['STATE_DIM'][0] - 1)
                cc = np.clip(cc, 0, config['STATE_DIM'][1] - 1)
                self.canvas[rr, cc] = 255

    def step(self, action):

        # do_nothing, q_line, q_curve, x0_line, y0_line, x1_line ,y1_line,
        # x0_c, y0_c, x1_c, y1_c, x2_c, y2_c, c
        self.t = self.t + 1
        if self.t == self.max_T - 1:
            self.terminal = True

        self.draw(action)

        if self.terminal and self.stroke_count == 0:
            reward = -100
        else:
            reward = self.find_reward()

        return self.get_state(), reward, self.terminal, {}

    def find_reward(self, n=16):
        if self.terminal:
            return self.classifier.get_score(self.get_state().reshape(-1, self.dim[0], self.dim[1], 1))

        # Roll-out
        canvas_current = self.canvas
        r = 0
        for i in range(n):
            self.canvas = canvas_current
            for tau in range(self.t + 1, self.max_T):
                _a = self.policy.step(self.get_state().reshape(-1, self.dim[0], self.dim[1], 1)) + np.random.normal(0, 1)
                self.draw(_a)
            r = r + self.classifier.get_score(self.get_state().reshape(-1, self.dim[0], self.dim[1], 1)) / n

        self.canvas = canvas_current
        return r

    def reset(self):
        self.canvas = np.zeros(self.dim)
        self.stroke_count = 0
        self.t = 0
        self.terminal = False

        return self.get_state()

    def render(self, mode='human', close=False):
        plt.imshow(self.canvas.reshape(28, 28))
        plt.show()

    def get_state(self):
        if self.stroke_count==0:
            #rr = np.random.randint(5, 20, size=2)
            #cc = np.random.randint(5, 20, size=2)
            return np.random.uniform(0, 0.1, self.canvas.shape)
        else:
            return self.canvas/255


def move_point(x, y):
    x_old, y_old = x, y
    x = x_old + np.random.randint(-3, 4)
    y = y_old + np.random.randint(-3, 4)

    while True:
        if x <= 0 or x >= config['STATE_DIM'][0] or x == x_old:
            x = x_old + np.random.randint(-1, 1)
        else:
            break

    while True:
        if y <= 0 or y >= config['STATE_DIM'][1] or y == y_old:
            y = y_old + np.random.randint(-1, 1)
        else:
            break

    return x, y


def scale(x, old_min, old_max, new_min=0, new_max=1):
    x = np.array(x)
    return new_min + ((x - old_min)*(new_max - new_min))/(old_max - old_min)
