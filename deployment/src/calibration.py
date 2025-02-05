import numpy as np

T_lidar_to_pixel = np.array([ # 2025-01-23 afternoon (hot glue)
    308.60880, -371.43606, -18.16256, 73.28075,
    232.43691, 7.92877, -388.54405, 61.57596,
    0.99869, 0.01813, -0.04782, 0.07770
], dtype=np.float32).reshape(3, 4)