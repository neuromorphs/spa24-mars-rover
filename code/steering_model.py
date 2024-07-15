import numpy as np


def skid_steer(speed,yaw_rate, wheel_base=0.381, wheel_radius=0.195/2):
    '''
    Default robot parameters taken from Pioneer 3DX datasheet:
    https://www.generationrobots.com/media/Pioneer3DX-P3DX-RevA.pdf

    Skid steer model from: 
    https://static1.squarespace.com/static/542ddec8e4b0158794bd1036/t/5a80e54cf9619ab1a97d1a04/1518396759609/Journal1.pdf (thanks Al!)

    '''

    kin_mat = np.array([
        [wheel_radius / 2, wheel_radius / 2],
        [- wheel_radius / wheel_base, wheel_radius / wheel_base]
        ])

    cmd_vec = np.array([[speed], [yaw_rate]])
    return np.linalg.solve(kin_mat, cmd_vec)

