import numpy as np
import cv2
size = 128, 128, 3
duration = 100
fps = 1
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
for _ in range(fps * duration):
    data = np.random.randint(0, 256, size, dtype='uint8')
    out.write(data)
out.release()