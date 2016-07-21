
import numpy as np
import random
import math

class Cannon:
    # A little ball trajectory simulation like the old Cannon games.
    # The data it generates can either be in the form of pixel images
    # of size 28x28 (like mnist!) or position of ball as two numbers 
    # between 0 and 1.

    def __init__(self):
        # ball starts at 0.5 and 0.0
        self.x_pos = .1428 
        self.y_pos = .5 
        self.x_vel = random.random()
        self.y_vel = 5*random.random() 
        # you can play with this
        self.grav = 0.0
        self.dt = .01 
        self.damp = 0.00 #probably set this to 0 for most applications
        # total steps
        self.total_steps = 0

    # very basic physics
    def restart(self):
        self.x_pos = .1428 
        self.y_pos = .5 
        self.x_vel = random.random()
        self.y_vel = 5*(random.random()-.5) 
        self.total_steps = 0

    # very basic physics
    def update_pos(self):
        for i in xrange(10):
            self.x_pos = self.x_pos + self.dt * self.x_vel
            self.y_pos = self.y_pos + self.dt * self.y_vel
            self.x_vel = self.x_vel + self.dt * (self.grav - (self.damp * self.x_vel))
            self.y_vel = self.y_vel + self.dt * self.damp * self.x_vel
            #self.y_vel = self.y_vel + self.dt * (self.grav - (self.damp * self.y_vel))
            # bounce
            if (0.1428 > self.x_pos):
                self.x_vel = -self.x_vel 
                self.x_pos = 0.1428 
            if ((1-0.1428) < self.x_pos):
                self.x_vel = -self.x_vel 
                self.x_pos = (1-0.1428)
            if (0.1428 > self.y_pos):
                self.y_vel = -self.y_vel 
                self.y_pos = 0.1428 
            if ((1-0.1428) < self.y_pos):
                self.y_vel = -self.y_vel 
                self.y_pos = (1-0.1428) 
        self.total_steps = self.total_steps + 1
 
    # generate pixell images
    def image_28x28(self):
        # same algorith as seen on
        # https://en.wikipedia.org/wiki/Midpoint_circle_algorithm
        im = np.zeros((28,28))
        radius = 4
        x0 = (self.x_pos * 28) // 1
        y0 = (self.y_pos * 28) // 1
        x = radius  
        y = 0
        decisionOver2 = 1 - x
        while(y <= x):
            im[int(x+x0) % 28, int(y+y0) % 28] = 1.0
            im[int(y+x0) % 28, int(x+y0) % 28] = 1.0
            im[int(-x+x0) % 28, int(y+y0) % 28] = 1.0
            im[int(-y+x0) % 28, int(x+y0) % 28] = 1.0
            im[int(-x+x0) % 28, int(-y+y0) % 28] = 1.0
            im[int(-y+x0) % 28, int(-x+y0) % 28] = 1.0
            im[int(x+x0) % 28, int(-y+y0) % 28] = 1.0
            im[int(y+x0) % 28, int(-x+y0) % 28] = 1.0
            y = y + 1
            if (decisionOver2 <=0):
                decisionOver2 = decisionOver2 + 2*y + 1
            else:
                x = x - 1
                decisionOver2 = decisionOver2 + 2*(y-x) + 1

        return im 

    def generate_28x28x4(self, num_steps, frame_num):
        if self.total_steps > 200:
            self.restart()
     
        xs = np.zeros([num_steps, 28, 28, 4])
        x = np.zeros([28, 28, 4])
        for s in xrange(num_steps):
          if s == 0:
            for i in xrange(frame_num):
              x[:,:,i] = self.image_28x28()
              self.update_pos()
          else:
            x[:,:,0:frame_num-1] = x[:,:,1:frame_num]
            x[:,:,frame_num-1] = self.image_28x28()
            self.update_pos()
          xs[s, :, :, :] = x[:,:,:]
 
        return xs


    def depricated_generate_28x28x4(self, batch_size, num_steps):
        cut_length = num_steps
        if num_steps < 5:
          cut_length = num_steps
          num_steps = 5

        x = np.zeros([batch_size, num_steps, 28, 28, 4])
        self.restart()
        for i in xrange(batch_size):
            #self.restart()
            x[i, 0, :, :, 0] = self.image_28x28()
            self.update_pos()
            x[i, 0, :, :, 1] = self.image_28x28()
            x[i, 1, :, :, 0] = x[i, 0, :, :, 1] 
            self.update_pos()
            x[i, 0, :, :, 2] = self.image_28x28()
            x[i, 1, :, :, 1] = x[i, 0, :, :, 2] 
            x[i, 2, :, :, 0] = x[i, 0, :, :, 2] 
            for j in xrange(num_steps-3):
                self.update_pos()
                x[i, j, :, :, 3] = self.image_28x28()
                x[i, j+1, :, :, 2] = x[i, j, :, :, 3] 
                x[i, j+2, :, :, 1] = x[i, j, :, :, 3] 
                x[i, j+3, :, :, 0] = x[i, j, :, :, 3] 
            self.update_pos()
            x[i, num_steps-3, :, :, 3] = self.image_28x28()
            x[i, num_steps-2, :, :, 2] = x[i, num_steps-3, :, :, 3] 
            x[i, num_steps-1, :, :, 1] = x[i, num_steps-3, :, :, 3] 
            self.update_pos()
            x[i, num_steps-2, :, :, 3] = self.image_28x28()
            x[i, num_steps-1, :, :, 2] = x[i, num_steps-2, :, :, 3] 
            self.update_pos()
            x[i, num_steps-1, :, :, 3] = self.image_28x28()

        x = x[:, 0:cut_length, :, :, :]
            
        return x

    def speed(self):
        return math.sqrt(self.x_vel ** 2 + self.y_vel ** 2)

if __name__ == "__main__":
    k = Cannon()




