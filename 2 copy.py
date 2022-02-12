from utils.vis import draw_skeleton, show3Dpose, draw_skeleton17, show3Dray, show3Dpose17_2d, show2Dpose17_3d, draw_ray, show3Dpose17_ray
from utils.functions import *
import matplotlib.pyplot as plt 
import numpy as np

from data import view0_joints, view1_joints, view2_joints, view3_joints
from data import view0_cano_3d_joints, view1_cano_3d_joints, view2_cano_3d_joints, view3_cano_3d_joints, pred_rot


def ray_equ(ax, p2_3d, p3d, sample_num=6, joints=(0,1),drawing=True):
    p3d = p3d.T
    p2_3d = p2_3d.T
    tt = np.linspace(0,sample_num,4)
    ray_sample_list = []
    for i in joints:     # 원래는 joint 개수별로 다
        x = [p2_3d[0][i] + t*(p3d[0][i] - p2_3d[0][i]) for t in tt] 
        y = [p2_3d[1][i] + t*(p3d[1][i] - p2_3d[1][i]) for t in tt] 
        z = [p2_3d[2][i] + t*(p3d[2][i] - p2_3d[2][i]) for t in tt] 
        xyz = np.array([x,y,z]).T 
        ray_sample_list.append(xyz)

    if drawing:
        for i in range(len(ray_sample_list)):
            x_set = np.array(ray_sample_list[i].T[0])  
            y_set = np.array(ray_sample_list[i].T[1])   
            z_set = np.array(ray_sample_list[i].T[2])  
            ax.plot(x_set, z_set, -y_set, lw=0.5, c='purple')
    
    ray_sample = np.transpose(np.array(ray_sample_list),(0,2,1))
    return ray_sample

# relative_rotation

relative_rotations_list = []
all_cams = ['cam0', 'cam1', 'cam2', 'cam3']
pred_rot_rs = pred_rot.reshape(-1, len(all_cams), 3, 3)  
for c_cnt in range(len(all_cams)):
    ## view consistency
    # get all cameras and active cameras
    ac = np.array(range(len(all_cams)))
    coi = np.delete(ac, c_cnt)     
    # ([b, 3(나머지3개view), 3, 3]) * ([b, 1(하나view), 3, 3]) = ([b, 3, 3, 3])
    relative_rotations_list.append(pred_rot[coi] @ np.transpose(pred_rot[[c_cnt]],(0, 2, 1)))

relative_rotations_list = np.array(relative_rotations_list)

# 이미지 
image_size = (1000,1000)
dummy_img = np.zeros((image_size[0], image_size[1], 3), np.uint8)

fig = plt.figure(figsize=(16,8))

# =====================================================================================
ax1 = fig.add_subplot(4,4,1)
draw_skeleton(view0_joints, dummy_img.copy(), ax1, data_type='h36m')
ax1 = fig.add_subplot(4,4,2)
draw_skeleton(view1_joints, dummy_img.copy(), ax1, data_type='h36m')
ax1 = fig.add_subplot(4,4,5)
draw_skeleton(view2_joints, dummy_img.copy(), ax1, data_type='h36m')
ax1 = fig.add_subplot(4,4,6)
draw_skeleton(view3_joints, dummy_img.copy(), ax1, data_type='h36m')
# =====================================================================================
ax2 = fig.add_subplot(2,2,2, projection='3d', aspect='auto')
show3Dpose(view0_cano_3d_joints, ax2, data_type='h36m', radius=1, lcolor='black',angles=(20,-60))
show3Dpose(view1_cano_3d_joints, ax2, data_type='h36m', radius=1, lcolor='blue',angles=(20,-60))
show3Dpose(view2_cano_3d_joints, ax2, data_type='h36m', radius=1, lcolor='red',angles=(20,-60))
show3Dpose(view3_cano_3d_joints, ax2, data_type='h36m', radius=1, lcolor='green',angles=(20,-60))
# =====================================================================================

# 각 view별 3d pose 
v0_3dp_1 = (pred_rot[0] @ (view0_cano_3d_joints.copy() - np.array([0,0,1])).T).T
v1_3dp_1 = (pred_rot[1] @ (view1_cano_3d_joints.copy() - np.array([0,0,1])).T).T
v2_3dp_1 = (pred_rot[2] @ (view2_cano_3d_joints.copy() - np.array([0,0,1])).T).T
v3_3dp_1 = (pred_rot[3] @ (view3_cano_3d_joints.copy() - np.array([0,0,1])).T).T

v0_3dp = (pred_rot[0] @ (view0_cano_3d_joints.copy()).T).T
v1_3dp = (pred_rot[1] @ (view1_cano_3d_joints.copy()).T).T
v2_3dp = (pred_rot[2] @ (view2_cano_3d_joints.copy()).T).T
v3_3dp = (pred_rot[3] @ (view3_cano_3d_joints.copy()).T).T

v0_rnorm_2dp, v0_2d_rnorm, v0_rjoints_2d = regular_normalized2d(view0_joints.copy()) 
v1_rnorm_2dp, v1_2d_rnorm, v1_rjoints_2d = regular_normalized2d(view1_joints.copy()) 
v2_rnorm_2dp, v2_2d_rnorm, v2_rjoints_2d = regular_normalized2d(view2_joints.copy()) 
v3_rnorm_2dp, v3_2d_rnorm, v3_rjoints_2d = regular_normalized2d(view3_joints.copy()) 

homo_v0_rnorm_2dp = np.concatenate((v0_rnorm_2dp, np.array([[1]*len(v0_rnorm_2dp)]).T),axis=1)
homo_v1_rnorm_2dp = np.concatenate((v1_rnorm_2dp, np.array([[1]*len(v1_rnorm_2dp)]).T),axis=1)
homo_v2_rnorm_2dp = np.concatenate((v2_rnorm_2dp, np.array([[1]*len(v2_rnorm_2dp)]).T),axis=1)
homo_v3_rnorm_2dp = np.concatenate((v3_rnorm_2dp, np.array([[1]*len(v3_rnorm_2dp)]).T),axis=1)
# =====================================================================================
ax3 = fig.add_subplot(2,4,5, projection='3d', aspect='auto')

show3Dpose(v1_3dp, ax3, data_type='h36m', radius=1, lcolor='blue',angles=(20,-60))

show3Dpose(v1_3dp_1, ax3, data_type='h36m', radius=1, lcolor='blue',angles=(20,-60))
show3Dpose(v3_3dp_1, ax3, data_type='h36m', radius=1, lcolor='green',angles=(20,-60))
# =====================================================================================
ax4 = fig.add_subplot(2,4,6, projection='3d', aspect='auto')

v1_3dp = (pred_rot[1] @ (view1_cano_3d_joints).T).T
v3_3dp = (pred_rot[3] @ (view3_cano_3d_joints).T).T

# show3Dpose(v1_3dp, ax4, data_type='h36m', radius=1, lcolor='blue',angles=(20,-60))

show3Dpose(v3_3dp, ax4, data_type='h36m', radius=1, lcolor='green',angles=(20,-60))

show3Dpose(homo_v3_rnorm_2dp, ax4, data_type='h36m', radius=1, lcolor='green',angles=(20,-60))
ray_sample_list = ray_equ(ax4, homo_v3_rnorm_2dp, v3_3dp, sample_num=3, joints=(0,3,6,9),drawing=True)

# =====================================================================================
ax5 = fig.add_subplot(2,4,7, projection='3d', aspect='auto')
show3Dpose(v1_3dp, ax5, data_type='h36m', radius=1, lcolor='blue',angles=(20,-60))
show3Dpose(homo_v1_rnorm_2dp, ax5, data_type='h36m', radius=1, lcolor='blue',angles=(20,-60))

# 초록을 파랑으로 돌려보기 2d도 같이 : 그때 2d 모양 보기
# 왠지 초록에서 2d랑 3d랑 잇는 ray를 파랑으로 회전하고 prject해서 보면 파랑 2d에 걸칠거 같은 느낌?

rot_v3_v1_3dp = (relative_rotations_list[3][1] @ v3_3dp.T).T
rot_homo_v3_v1_rnorm_2dp = (relative_rotations_list[3][1] @ homo_v3_rnorm_2dp.T).T
show3Dpose(rot_v3_v1_3dp, ax5, data_type='h36m', radius=1, lcolor='green',angles=(20,-60))
show3Dpose(rot_homo_v3_v1_rnorm_2dp, ax5, data_type='h36m', radius=1, lcolor='green',angles=(20,-60))

rot_3to1_ray = np.transpose((relative_rotations_list[3][1] @ ray_sample_list),(0,2,1))
for i in range(len(rot_3to1_ray)):
    x_set = np.array(rot_3to1_ray[i].T[0])  
    y_set = np.array(rot_3to1_ray[i].T[1])   
    z_set = np.array(rot_3to1_ray[i].T[2])  
    ax5.plot(x_set, z_set, -y_set, lw=0.5, c='purple')


ax6 = fig.add_subplot(2,4,8, projection='3d', aspect='auto')
#show3Dpose(v1_3dp, ax6, data_type='h36m', radius=1, lcolor='blue',angles=(20,-60))
show3Dpose(homo_v1_rnorm_2dp, ax6, data_type='h36m', radius=1, lcolor='blue',angles=(20,-60))

# 초록을 파랑으로 돌려보기 2d도 같이 : 그때 2d 모양 보기
# 왠지 초록에서 2d랑 3d랑 잇는 ray를 파랑으로 회전하고 prject해서 보면 파랑 2d에 걸칠거 같은 느낌?

rot_v3_v1_3dp = (relative_rotations_list[3][1] @ v3_3dp.T).T
rot_homo_v3_v1_rnorm_2dp = (relative_rotations_list[3][1] @ homo_v3_rnorm_2dp.T).T
#show3Dpose(rot_v3_v1_3dp, ax6, data_type='h36m', radius=1, lcolor='green',angles=(20,-60))
show3Dpose(rot_homo_v3_v1_rnorm_2dp, ax6, data_type='h36m', radius=1, lcolor='green',angles=(20,-60))

rot_3to1_ray = np.transpose((relative_rotations_list[3][1] @ ray_sample_list),(0,2,1))
for i in range(len(rot_3to1_ray)):
    x_set = np.array(rot_3to1_ray[i].T[0])  
    y_set = np.array(rot_3to1_ray[i].T[1])   
    z_set = np.array(rot_3to1_ray[i].T[2])  
    ax6.plot(x_set, z_set, -y_set, lw=0.5, c='purple')


fig.tight_layout()
plt.show()