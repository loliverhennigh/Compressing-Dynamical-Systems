

import bouncing_balls as b
import cv2
import numpy as np

res = 84 
n_balls = 4
T = 200
dat = b.bounce_vec(res,n_balls,T)
b.show_V(dat)




print(dat.shape)

dat = dat.reshape(T, res, res) 
dat = np.uint8(np.abs(dat * 255))

fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') 
video = cv2.VideoWriter()
success = video.open("test.mov", fourcc, 4, (84, 84), True)


for i in xrange(T-3):
  frame = dat[i:i+3, :, :] 
  frame = np.transpose(frame, (1,2,0))
  print(frame.shape)
  video.write(frame)

video.release()
cv2.destroyAllWindows()




