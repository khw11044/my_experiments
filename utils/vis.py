import numpy as np
import cv2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

H36M_JOINTMAP = [
    [0,1],      
    [1,2],      #
    [2,3],      #
    [0,4],      
    [4,5],
    [5,6],
    [0,7],
    [7,8],
    [8,9],
    [7,10],
    [7,13],
    [10,11],
    [11,12],
    [13,14],
    [14,15]
    ]

H36M_JOINTMAP_17 = [        # 0: root / 1 : Rhip / 2: Rknee / 3: RAnkle / 4: Lhip / 5: Lknee / 6: LAnkle
    [0,1],      
    [1,2],      #
    [2,3],      #
    [0,4],      
    [4,5],
    [5,6],
    [0,7],
    [7,8],
    [8,9],
    [9,10],
    [8,11],
    [8,14],
    [11,12],
    [12,13],
    [14,15],
    [15,16]
    ]

MPII_JOINTMAP = [
    [0,1],      
    [1,2],      #
    [3,4],      #
    [4,5],      
    [6,0],
    [6,3],
    [6,7],
    [7,8],
    [8,9],
    [7,10],
    [7,13],
    [10,11],
    [11,12],
    [13,14],
    [14,15]
    ]

def draw_skeleton(annot2d_keypoints, img, ax, data_type='h36m'):
    if data_type == 'h36m':
        JOINTMAP = H36M_JOINTMAP
        root_joint_numer = 0
    elif data_type == 'h36m17':
        JOINTMAP = H36M_JOINTMAP_17
    else:
        JOINTMAP = MPII_JOINTMAP
        root_joint_numer = 6


    # 빈 이미지(검정 바탕)에 스켈레톤 그리기 
    for j in range(len(JOINTMAP)):
        child = tuple(np.array(annot2d_keypoints[JOINTMAP[j][0]][:2]).astype(int))
        parent = tuple(np.array(annot2d_keypoints[JOINTMAP[j][1]][:2]).astype(int))
        color = (144, 243, 34)

        cv2.circle(img, parent, 8, (255, 255, 255), -1)
        cv2.line(img, child, parent, color, 3) 


    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def draw_skeleton17(annot2d_keypoints, img, ax, data_type='h36m', color=(0,0,255)):
    if data_type == 'h36m':
        JOINTMAP = H36M_JOINTMAP
        root_joint_numer = 0
    elif data_type == 'h36m17':
        JOINTMAP = H36M_JOINTMAP_17
    else:
        JOINTMAP = MPII_JOINTMAP
        root_joint_numer = 6


    # 빈 이미지(검정 바탕)에 스켈레톤 그리기 
    for j in range(len(JOINTMAP)):
        child = tuple(np.array(annot2d_keypoints[JOINTMAP[j][0]][:2]).astype(int))
        parent = tuple(np.array(annot2d_keypoints[JOINTMAP[j][1]][:2]).astype(int))
        lcolor = (144, 243, 34)

        cv2.circle(img, parent, 8, color, -1)
        cv2.line(img, child, parent, lcolor, 3) 


    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def show3Dpose(vals, ax, radius=40, data_type='h36m', lcolor='red', rcolor='#0000ff',angles=(10,-60)):
    if data_type == 'h36m':
        JOINTMAP = H36M_JOINTMAP
        root_joint_numer = 0
    elif data_type == 'h36m17':
        JOINTMAP = H36M_JOINTMAP_17
    else:
        JOINTMAP = MPII_JOINTMAP
        root_joint_numer = 6

    xroot, yroot, zroot = vals[root_joint_numer, 0], vals[root_joint_numer, 1], vals[root_joint_numer, 2]

    for ind, (i,j) in enumerate(JOINTMAP):
        x, y, z = [np.array([vals[i, c], vals[j, c]]) for c in range(3)]
        # x -= xroot
        # y -= yroot
        # z -= zroot
        ax.plot(x, z, -y, lw=2, c=lcolor)

    RADIUS = radius  # space around the subject


    ax.set_xlim3d([-RADIUS, RADIUS])
    ax.set_ylim3d([-RADIUS, RADIUS])
    ax.set_zlim3d([-RADIUS, RADIUS])
    ax.view_init(angles[0], angles[-1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

def show3Dpose17(vals, ax, radius=40, data_type='h36m', lcolor='red', rcolor='#0000ff',angles=(10,-60)):
    if data_type == 'h36m':
        JOINTMAP = H36M_JOINTMAP
        root_joint_numer = 0
    elif data_type == 'h36m17':
        JOINTMAP = H36M_JOINTMAP_17
    else:
        JOINTMAP = MPII_JOINTMAP
        root_joint_numer = 6

    xroot, yroot, zroot = vals[root_joint_numer, 0], vals[root_joint_numer, 1], vals[root_joint_numer, 2]

    for ind, (i,j) in enumerate(JOINTMAP):
        x, y, z = [np.array([vals[i, c], vals[j, c]]) for c in range(3)]
        # x -= xroot
        # y -= yroot
        # z -= zroot
        ax.plot(x, z, -y, lw=2, c=lcolor)

    RADIUS = radius  # space around the subject


    ax.set_xlim3d([-RADIUS, RADIUS])
    ax.set_ylim3d([-RADIUS, RADIUS])
    ax.set_zlim3d([-RADIUS, RADIUS])
    ax.view_init(angles[0], angles[-1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

def show3Dpose17_2d(vals, ax, radius=40, data_type='h36m', lcolor='red',angles=(10,-60)):
    if data_type == 'h36m':
        JOINTMAP = H36M_JOINTMAP_17
        root_joint_numer = 0
    else:
        JOINTMAP = MPII_JOINTMAP
        root_joint_numer = 6

    xroot, yroot, zroot = vals[root_joint_numer, 0], vals[root_joint_numer, 1], vals[root_joint_numer, 2]

    for ind, (i,j) in enumerate(JOINTMAP):
        x, y, z = [np.array([vals[i, c], vals[j, c]]) for c in range(3)]
        # x -= xroot
        # y -= yroot
        # z -= zroot
        z = np.array([0,0])
        ax.plot(x, z, -y, lw=2, c=lcolor)
    
    RADIUS = radius  # space around the subject


    ax.set_xlim3d([-RADIUS, RADIUS])
    ax.set_ylim3d([-RADIUS, RADIUS])
    ax.set_zlim3d([-RADIUS, RADIUS])
    ax.view_init(angles[0], angles[-1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

def show2Dpose17_3d(vals, ax, depth, radius=40, data_type='h36m', lcolor='red',angles=(10,-60)):
    if data_type == 'h36m':
        JOINTMAP = H36M_JOINTMAP_17
        root_joint_numer = 0
    else:
        JOINTMAP = MPII_JOINTMAP
        root_joint_numer = 6

    for ind, (i,j) in enumerate(JOINTMAP):
        x, y = [np.array([vals[i, c], vals[j, c]]) for c in range(2)]
        z = np.array([depth,depth])
        ax.plot(x, z, -y, lw=2, c=lcolor)
    
    RADIUS = radius  # space around the subject


    ax.set_xlim3d([-RADIUS, RADIUS])
    ax.set_ylim3d([-RADIUS, RADIUS])
    ax.set_zlim3d([-RADIUS, RADIUS])
    ax.view_init(angles[0], angles[-1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    pass

def show3Dray(vals, ax, radius=40, data_type='h36m', lcolor='red', rcolor='#0000ff',angles=(10,-60)):
    if data_type == 'h36m':
        JOINTMAP = H36M_JOINTMAP_17
        root_joint_numer = 0
    else:
        JOINTMAP = MPII_JOINTMAP
        root_joint_numer = 6
    
    # vals = vals[6:9]
    # for ind, (x,y) in enumerate(vals):
    #     x_set = np.array([0,x])  
    #     y_set =np.array([0,y])  
    #     z_set = np.array([1,0])  
    #     ax.plot(x_set, z_set, -y_set, lw=2, c=lcolor)

    # for ind, (x,y) in enumerate(vals):
    #     x_set = np.array([x,x])  
    #     y_set =np.array([y,y])  
    #     z_set = np.array([-2,2])  
    #     ax.plot(x_set, z_set, -y_set, lw=2, c=lcolor)
    start_val = vals[:17]
    middle_val = vals[17:34]
    end_val = vals[34:]

    for i in range(len(start_val)):
        x_set = np.array([start_val[i][0],middle_val[i][0],end_val[i][0]])  
        y_set = np.array([start_val[i][1],middle_val[i][1],end_val[i][1]])  
        z_set = np.array([start_val[i][2],middle_val[i][2],end_val[i][2]])  
        ax.plot(x_set, z_set, -y_set, lw=2, c=lcolor)
    
    RADIUS = radius  # space around the subject


    ax.set_xlim3d([-RADIUS, RADIUS])
    ax.set_ylim3d([-RADIUS, RADIUS])
    ax.set_zlim3d([-RADIUS, RADIUS])
    ax.view_init(angles[0], angles[-1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def draw_ray(camera_centre, vals, ax, radius=40, data_type='h36m', lcolor='red', rcolor='#0000ff',angles=(10,-60)):
    if data_type == 'h36m':
        JOINTMAP = H36M_JOINTMAP_17

    else:
        JOINTMAP = MPII_JOINTMAP
    
    # 카메라가 (0,0,-10) 위치에 있고 point가 (a,b,0) 이면 
    # x = ta
    # y = tb
    # z = -10 + 10t
    t = 2
    for i in range(len(vals)):
        # x_set = np.array([camera_centre[0],vals[i][0]])  
        # y_set = np.array([camera_centre[1],vals[i][1]])  
        # z_set = np.array([camera_centre[2],0])  
        x_set = np.array([camera_centre[0],vals[i][0]])  
        y_set = np.array([camera_centre[1],vals[i][1]])  
        z_set = np.array([camera_centre[2],0])  
        ax.plot(x_set, z_set, -y_set, lw=2, c=lcolor)
    
    RADIUS = radius  # space around the subject


    ax.set_xlim3d([-RADIUS, RADIUS])
    ax.set_ylim3d([-RADIUS, RADIUS])
    ax.set_zlim3d([-RADIUS, RADIUS])
    ax.view_init(angles[0], angles[-1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def show3Dpose17_ray(vals, ax, radius=40, data_type='h36m', lcolor='red', rcolor='#0000ff',angles=(10,-60)):

    if data_type == 'h36m':
        JOINTMAP = H36M_JOINTMAP_17
        root_joint_numer = 0
    else:
        JOINTMAP = MPII_JOINTMAP
        root_joint_numer = 6

    xroot, yroot, zroot = vals[root_joint_numer, 0], vals[root_joint_numer, 1], vals[root_joint_numer, 2]

    for ind, (i,j) in enumerate(JOINTMAP):
        x, y, z = [np.array([vals[i, c], vals[j, c]]) for c in range(3)]
        # x -= xroot
        # y -= yroot
        # z -= zroot
        ax.plot(x, z, -y, lw=2, c=lcolor)

    RADIUS = radius  # space around the subject


    ax.set_xlim3d([-3, 3])
    ax.set_ylim3d([-RADIUS, 2-RADIUS])
    ax.set_zlim3d([-3, 3])
    ax.view_init(angles[0], angles[-1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")