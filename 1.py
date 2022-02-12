from utils.vis import draw_skeleton, show3Dpose, draw_skeleton17, show3Dray, show3Dpose17_2d, show2Dpose17_3d, draw_ray, show3Dpose17_ray
from utils.functions import *
import matplotlib.pyplot as plt 
import numpy as np

from data import view0_joints, view1_joints, view2_joints, view3_joints
from data import view0_cano_3d_joints, view1_cano_3d_joints, view2_cano_3d_joints, view3_cano_3d_joints, pred_rot


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
ax2 = fig.add_subplot(2,4,5, projection='3d', aspect='auto')
show3Dpose(view0_cano_3d_joints, ax2, data_type='h36m', radius=1, lcolor='black',angles=(20,-60))
show3Dpose(view1_cano_3d_joints, ax2, data_type='h36m', radius=1, lcolor='blue',angles=(20,-60))
show3Dpose(view2_cano_3d_joints, ax2, data_type='h36m', radius=1, lcolor='red',angles=(20,-60))
show3Dpose(view3_cano_3d_joints, ax2, data_type='h36m', radius=1, lcolor='green',angles=(20,-60))
# =====================================================================================

# 각 view별 3d pose 
v0_3dp = (pred_rot[0] @ view0_cano_3d_joints.T).T
v1_3dp = (pred_rot[1] @ view1_cano_3d_joints.T).T
v2_3dp = (pred_rot[2] @ view2_cano_3d_joints.T).T
v3_3dp = (pred_rot[3] @ view3_cano_3d_joints.T).T


v0_rnorm_2dp, v0_2d_rnorm, v0_rjoints_2d = regular_normalized2d(view0_joints.copy()) 
v1_rnorm_2dp, v1_2d_rnorm, v1_rjoints_2d = regular_normalized2d(view1_joints.copy()) 
v2_rnorm_2dp, v2_2d_rnorm, v2_rjoints_2d = regular_normalized2d(view1_joints.copy()) 
v3_rnorm_2dp, v3_2d_rnorm, v3_rjoints_2d = regular_normalized2d(view1_joints.copy()) 


# =====================================================================================
ax3 = fig.add_subplot(2,4,6, projection='3d', aspect='auto')
show3Dpose(v0_3dp, ax3, data_type='h36m', radius=1, lcolor='black',angles=(20,-60))
show3Dpose(v1_3dp, ax3, data_type='h36m', radius=1, lcolor='blue',angles=(20,-60))
show3Dpose(v2_3dp, ax3, data_type='h36m', radius=1, lcolor='red',angles=(20,-60))
show3Dpose(v3_3dp, ax3, data_type='h36m', radius=1, lcolor='green',angles=(20,-60))


fig.tight_layout()
plt.show()