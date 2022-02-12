import numpy as np

def scaled_normalized2d(pose):
    scale_p2d = np.sqrt(np.square(pose.T.reshape(-1,32)[:, 0:32]).sum(axis=1) / 32)
    p2d_scaled = pose.T.reshape(-1,32)[:, 0:32] / scale_p2d                             # scale_p2d : fx 이게 focal length
    norm_scaled_p2d = p2d_scaled[0].reshape(2,16).T
    return norm_scaled_p2d, scale_p2d

def scaled_normalized3d_2d(pose):   # projection
    scale_p3d = np.sqrt(np.square(pose.T.reshape(-1,48)[:, 0:32]).sum(axis=1) / 32)  # root(1 / 34)  # projection 
    p3d_scaled = pose.T.reshape(-1,48)[:, 0:32] / scale_p3d                             # scale_p3d 이게 focal length이자 K
    norm_scaled_projected_p3d = p3d_scaled[0].reshape(2,17).T
    return norm_scaled_projected_p3d, scale_p3d

def scaled_normalized3d(pose):   # projection
    scale_p3d = np.sqrt(np.square(pose.T.reshape(-1,48)[:, 0:32]).sum(axis=1) / 32)    # projection
    p3d_scaled = pose.T.reshape(-1,48)[:, 0:48] / scale_p3d
    norm_scaled_p3d = p3d_scaled[0].reshape(3,16).T
    return norm_scaled_p3d, scale_p3d

def regular_normalized3d(poseset):
    pose_norm_list = []
    poseset = poseset.reshape(-1,16,3)

    for i in range(len(poseset)):
        root_joints = poseset[i].T[:, [0]]                                     
        pose_norm = np.linalg.norm((poseset[i].T - root_joints).reshape(-1, 48), ord=2, axis=1, keepdims=True)  
        poseset[i] = (poseset[i].T - root_joints).T                
        poseset[i] /= pose_norm
        pose_norm_list.append(pose_norm)

    return poseset[0], np.array(pose_norm_list), root_joints


def regular_normalized2d(poseset):
    pose_norm_list = []
    poseset = poseset.reshape(-1,16,2)

    for i in range(len(poseset)):
        root_joints = poseset[i].T[:, [0]]                                     
        pose_norm = np.linalg.norm((poseset[i].T - root_joints).reshape(-1, 32), ord=2, axis=1, keepdims=True)  
        poseset[i] = (poseset[i].T - root_joints).T                
        poseset[i] /= pose_norm
        pose_norm_list.append(pose_norm)

    return poseset[0], np.array(pose_norm_list), root_joints


# 0-centered normalize_scaling
def normalize_scale_2d(pose2d):
    norm_2d_pose, norm_2d, root_joints_2d = regular_normalized2d(pose2d.copy())
    norm_scaled_p2d, scale_p2d = scaled_normalized2d(norm_2d_pose)
    return norm_scaled_p2d, scale_p2d, norm_2d, root_joints_2d


def pose_norm_K(homogen_pose):
    pose = homogen_pose[:,:2]
    root_joint = pose.T[:, [0]]
    f = np.linalg.norm((pose.T - root_joint).reshape(-1, 32), ord=2, axis=1, keepdims=True)[0][0]
    K = np.array([[f,0,root_joint[0][0]],[0,f,root_joint[1][0]],[0,0,1]])
    return K

def ray_equ(ax, cam, p3d, sample_num=6, joints=(0,1),drawing=True):
    p3d = p3d.T
    tt = np.linspace(0,sample_num,4)
    ray_sample_list = []
    for i in joints:     # 원래는 joint 개수별로 다
        x = [cam[0] + t*(p3d[0][i] - cam[0]) for t in tt] 
        y = [cam[1] + t*(p3d[1][i] - cam[1]) for t in tt] 
        z = [cam[2] + t*(p3d[2][i] - cam[2]) for t in tt] 
        xyz = np.array([x,y,z]).T 
        ray_sample_list.append(xyz)

    if drawing:
        for i in range(len(ray_sample_list)):
            x_set = np.array(ray_sample_list[i].T[0])  
            y_set = np.array(ray_sample_list[i].T[1])   
            z_set = np.array(ray_sample_list[i].T[2])  
            ax.plot(x_set, z_set, -y_set, lw=2, c='purple')
    
    return np.array(ray_sample_list)