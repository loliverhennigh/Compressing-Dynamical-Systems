



import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 

import cannon as cn


writer = animation.writers['ffmpeg'](fps=1)

seq_length = 4
batch_size = 10

k = cn.Cannon()
x = k.generate_28x28x4(batch_size,seq_length)

ims_generated = []
fig = plt.figure()

for i in xrange(batch_size):
  for j in xrange(seq_length):
    for k in xrange(4):
      print(i)
      print(j)
      print(k)
      print(x.shape)
      new_im = x[i,j,:,:,k]
      ims_generated.append((plt.imshow(new_im),))
m_ani = animation.ArtistAnimation(fig, ims_generated, interval= 5000, repeat_delay=3000, blit=True)
m_ani.save("test.mp4", writer=writer)
       
