from Utils.BVH_loader import load as bvh_load
from Utils.BVH_loader import save as bvh_save

# data = bvh_load(r"F:\Semantic-Gesticulator-Official\001_Neutral_0_x_1_0_retarget2mocap.bvh")
# # print(data.joint_names, len(data.joint_names))
# body_joints = [0,1,2,3,5,6,7,9,10,11,12,13,14,16,17,18,19,44,45,46,47]
# hand_joints = [20,21,22,24,25,26,27,29,30,31,32,34,35,36,37,39,40,41,42,
#                48,49,50,52,53,54,55,57,58,59,60,62,63,64,65,67,68,69,70]
# print(len(body_joints))
# for idx in body_joints:
#     print(data.joint_names[idx])

data = bvh_load(r"../Data/JIYI_Data/jiyi_data/CAREER_1_4_001_bar_0.bvh")
print(data.joint_names, len(data.joint_names))
body_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 93, 94, 104, 105, 313, 314, 324, 325, 592, 593, 594, 638, 639, 640]
hand_joints = [107, 108, 109, 144, 145, 146, 180, 181, 182, 210, 211, 212, 247, 248, 249, 327,
        328, 329, 364, 365, 366, 400, 401, 402, 430, 431, 432, 467, 468, 469]

eval_list =['pelvis', 'spine_01', 'spine_02', 'spine_03', 'spine_04', 'spine_05', 'neck_01', 'neck_02', 'head', 'Back_Hair_0', 'Back_Hair_1', 'Back_Hair_2', 'Back_Hair_3', 'Back_Hair_4', 'Back_Hair_5', 'Back_Hair_6', 'Back_Hair_7', 'Back_Hair_8', 'R_hair_0', 'R_hair_1', 'R_hair_2', 'R_hair_3', 'R1_hair_0', 'R1_hair_1', 'R1_hair_2', 'R1_hair_3', 'R1_hair_4', 'R2_hair_0', 'R2_hair_1', 'R2_hair_2', 'Front_A_hair_0', 'Front_A_hair_1', 'Front_A_hair_2', 'Front_A_hair_3', 'Front_B_hair_0', 'Front_B_hair_1', 'Front_B_hair_2', 'Front_B_hair_3', 'Front_B_hair_4', 'Front_B_hair_5', 'Front_B_hair_6', 'Front_C_hair_0', 'Front_C_hair_1', 'Front_C_hair_2', 'Front_C_hair_3', 'Front_C_hair_4', 'Front_C_hair_5', 'Front_C_hair_6', 'Front_D_hair_0', 'Front_D_hair_1', 'Front_D_hair_2', 'Front_D_hair_3', 'Front_D_hair_4', 'Front_D_hair_5', 'Front_D_hair_6', 'L4_hair_0', 'L4_hair_1', 'L4_hair_2', 'L4_hair_3', 'L4_hair_4', 'L2_hair_0', 'L2_hair_1', 'L2_hair_2', 'L2_hair_3', 'L2_hair_4', 'L2_hair_5', 'L_hair_011', 'L_hair_012', 'L_hair_013', 'L_hair_014', 'L1_hair_0', 'L1_hair_1', 'L1_hair_2', 'L1_hair_3', 'L1_hair_4', 'L1_hair_5', 'L1_hair_6', 'L1_hair_7', 'L3_hair_0', 'L3_hair_1', 'L3_hair_2', 'clavicle_l', 'upperarm_l', 'upperarm_correctiveRoot_l', 'upperarm_bck_l', 'upperarm_fwd_l', 'upperarm_in_l', 'upperarm_out_l', 'lowerarm_l', 'hand_l', 'pinky_metacarpal_l', 'pinky_01_l', 'pinky_02_l', 'pinky_03_l', 'pinky_03_bulge_l', 'pinky_03_half_l', 'pinky_03_in_l', 'pinky_02_dip_l', 'pinky_02_bulge_l', 'pinky_02_side_out_l', 'pinky_02_side_inn_l', 'pinky_02_half_l', 'pinky_02_in_l', 'pinky_02_pip_l', 'pinky_01_palmMid_l', 'pinky_01_bulge_l', 'pinky_01_side_out_l', 'pinky_01_side_inn_l', 'pinky_01_half_l', 'pinky_01_mcp_l', 'pinky_01_palm_l', 'pinky_metacarpal_slide_l', 'ring_metacarpal_l', 'ring_01_l', 'ring_02_l', 'ring_03_l', 'ring_03_bulge_l', 'ring_03_half_l', 'ring_02_dip_l', 'ring_03_in_l', 'ring_02_bulge_l', 'ring_02_side_out_l', 'ring_02_side_inn_l', 'ring_02_half_l', 'ring_02_in_l', 'ring_02_pip_l', 'ring_01_palmMid_l', 'ring_01_bulge_l', 'ring_01_side_out_l', 'ring_01_side_inn_l', 'ring_01_half_l', 'ring_01_mcp_l', 'ring_01_palm_l', 'ring_metacarpal_slide_l', 'thumb_01_l', 'thumb_02_l', 'thumb_03_l', 'thumb_03_bulge_l', 'thumb_03_side_out_l', 'thumb_03_side_inn_l', 'thumb_03_half_l', 'thumb_03_pip_l', 'thumb_03_in_l', 'thumb_02_bulge_l', 'thumb_02_side_out_l', 'thumb_02_side_inn_l', 'thumb_02_half_l', 'thumb_02_in_l', 'thumb_02_mcp_l', 'thumb_01_side_out_l', 'thumb_01_side_inn_l', 'middle_metacarpal_l', 'middle_01_l', 'middle_02_l', 'middle_03_l', 'middle_03_bulge_l', 'middle_03_half_l', 'middle_03_in_l', 'middle_02_dip_l', 'middle_02_bulge_l', 'middle_02_side_out_l', 'middle_02_side_inn_l', 'middle_02_half_l', 'middle_02_pip_l', 'middle_02_in_l', 'middle_01_palmMid_l', 'middle_01_bulge_l', 'middle_01_side_out_l', 'middle_01_side_inn_l', 'middle_01_half_l', 'middle_01_mcp_l', 'middle_01_palm_l', 'middle_metacarpal_slide_l', 'index_metacarpal_l', 'index_01_l', 'index_02_l', 'index_03_l', 'index_03_bulge_l', 'index_03_half_l', 'index_02_dip_l', 'index_03_in_l', 'index_02_bulge_l', 'index_02_side_out_l', 'index_02_side_inn_l', 'index_02_half_l', 'index_02_in_l', 'index_02_pip_l', 'index_01_palmMid_l', 'index_01_bulge_l', 'index_01_side_out_l', 'index_01_side_inn_l', 'index_01_half_l', 'index_01_mcp_l', 'index_01_palm_l', 'index_metacarpal_slide_l', 'wrist_inner_l', 'wrist_outer_l', 'lowerarm_twist_02_l', 'lowerarm_twist_01_l', 'lowerarm_correctiveRoot_l', 'lowerarm_in_l', 'lowerarm_out_l', 'lowerarm_fwd_l', 'lowerarm_bck_l', 'upperarm_twist_01_l', 'upperarm_twist_02_l', 'upperarm_tricep_l', 'upperarm_bicep_l', 'upperarm_twistCor_02_l', 'clavicle_out_l', 'clavicle_scap_l', 'clavicle_r', 'upperarm_r', 'upperarm_correctiveRoot_r', 'upperarm_bck_r', 'upperarm_in_r', 'upperarm_fwd_r', 'upperarm_out_r', 'lowerarm_r', 'hand_r', 'pinky_metacarpal_r', 'pinky_01_r', 'pinky_02_r', 'pinky_03_r', 'pinky_03_bulge_r', 'pinky_03_half_r', 'pinky_03_in_r', 'pinky_02_dip_r', 'pinky_02_bulge_r', 'pinky_02_side_out_r', 'pinky_02_side_inn_r', 'pinky_02_half_r', 'pinky_02_in_r', 'pinky_02_pip_r', 'pinky_01_palmMid_r', 'pinky_01_bulge_r', 'pinky_01_side_out_r', 'pinky_01_side_inn_r', 'pinky_01_half_r', 'pinky_01_mcp_r', 'pinky_01_palm_r', 'pinky_metacarpal_slide_r', 'ring_metacarpal_r', 'ring_01_r', 'ring_02_r', 'ring_03_r', 'ring_03_bulge_r', 'ring_03_half_r', 'ring_02_dip_r', 'ring_03_in_r', 'ring_02_bulge_r', 'ring_02_side_out_r', 'ring_02_side_inn_r', 'ring_02_half_r', 'ring_02_in_r', 'ring_02_pip_r', 'ring_01_palmMid_r', 'ring_01_bulge_r', 'ring_01_side_out_r', 'ring_01_side_inn_r', 'ring_01_half_r', 'ring_01_mcp_r', 'ring_01_palm_r', 'ring_metacarpal_slide_r', 'thumb_01_r', 'thumb_02_r', 'thumb_03_r', 'thumb_03_bulge_r', 'thumb_03_side_out_r', 'thumb_03_side_inn_r', 'thumb_03_half_r', 'thumb_03_pip_r', 'thumb_03_in_r', 'thumb_02_bulge_r', 'thumb_02_side_out_r', 'thumb_02_side_inn_r', 'thumb_02_half_r', 'thumb_02_in_r', 'thumb_02_mcp_r', 'thumb_01_side_out_r', 'thumb_01_side_inn_r', 'middle_metacarpal_r', 'middle_01_r', 'middle_02_r', 'middle_03_r', 'middle_03_bulge_r', 'middle_03_half_r', 'middle_03_in_r', 'middle_02_dip_r', 'middle_02_bulge_r', 'middle_02_side_out_r', 'middle_02_side_inn_r', 'middle_02_half_r', 'middle_02_pip_r', 'middle_02_in_r', 'middle_01_palmMid_r', 'middle_01_bulge_r', 'middle_01_side_out_r', 'middle_01_side_inn_r', 'middle_01_half_r', 'middle_01_mcp_r', 'middle_01_palm_r', 'middle_metacarpal_slide_r', 'index_metacarpal_r', 'index_01_r', 'index_02_r', 'index_03_r', 'index_03_bulge_r', 'index_03_half_r', 'index_02_dip_r', 'index_03_in_r', 'index_02_bulge_r', 'index_02_side_out_r', 'index_02_side_inn_r', 'index_02_half_r', 'index_02_in_r', 'index_02_pip_r', 'index_01_palmMid_r', 'index_01_bulge_r', 'index_01_side_out_r', 'index_01_side_inn_r', 'index_01_half_r', 'index_01_mcp_r', 'index_01_palm_r', 'index_metacarpal_slide_r', 'wrist_inner_r', 'wrist_outer_r', 'lowerarm_twist_02_r', 'lowerarm_twist_01_r', 'lowerarm_correctiveRoot_r', 'lowerarm_out_r', 'lowerarm_in_r', 'lowerarm_fwd_r', 'lowerarm_bck_r', 'upperarm_twist_01_r', 'upperarm_twist_02_r', 'upperarm_tricep_r', 'upperarm_bicep_r', 'upperarm_twistCor_02_r', 'clavicle_out_r', 'clavicle_scap_r', 'clavicle_pec_r', 'spine_04_latissimus_l', 'spine_04_latissimus_r', 'clavicle_pec_l', 'coat_R_03_jot', 'coat_R_02_jot', 'coat_R_01_jot', 'coat_B_jot', 'coat_L_03_jot', 'coat_L_02_jot', 'coat_L_01_jot', 'collar_B_jot', 'collar_R_01_jot', 'collar_R_02_jot', 'collar_L_01_jot', 'collar_L_02_jot', 'necklace_F_jot', 'nametag01_jot', 'nametag02_jot', 'bullet_jot', 'necklace_jot', 'necklace_R_jot', 'necklace_B_jot', 'necklace_L_jot', 'belt_jot', 'belt_F_jot', 'belt_R_jot', 'belt_L_jot', 'belt_B_jot', 'belt_bag1_jot', 'belt_bag3_jot', 'thigh_r', 'calf_r', 'foot_r', 'ball_r', 'littletoe_01_r', 'ringtoe_01_r', 'middletoe_01_r', 'bigtoe_01_r', 'indextoe_01_r', 'ankle_bck_r', 'ankle_fwd_r', 'calf_twist_02_r', 'calf_twist_01_r', 'calf_correctiveRoot_r', 'calf_kneeBack_r', 'calf_knee_r', 'kneepad_R_jot', 'thigh_twist_01_r', 'thigh_twist_02_r', 'thigh_correctiveRoot_r', 'thigh_fwd_r', 'thigh_bck_r', 'thigh_out_r', 'thigh_in_r', 'thigh_bck_lwr_r', 'thigh_fwd_lwr_r', 'thigh_l', 'calf_l', 'foot_l', 'ball_l', 'indextoe_01_l', 'bigtoe_01_l', 'littletoe_01_l', 'middletoe_01_l', 'ringtoe_01_l', 'ankle_bck_l', 'ankle_fwd_l', 'calf_twist_02_l', 'calf_twist_01_l', 'calf_correctiveRoot_l', 'calf_kneeBack_l', 'calf_knee_l', 'kneepad_L_jot', 'thigh_twist_01_l', 'thigh_twistCor_01_l', 'Legloop2_L_jot', 'Legloop2_R_jot', 'Legloop2_B_jot', 'Legloop2_F_jot', 'bag_jot', 'thigh_twist_02_l', 'thigh_correctiveRoot_l', 'thigh_bck_l', 'thigh_fwd_l', 'thigh_out_l', 'thigh_bck_lwr_l', 'thigh_in_l', 'thigh_fwd_lwr_l']


target_list = []
for body_idx in body_joints:
    ori_name = data.joint_names[body_idx]
    for target_idx, target_name in enumerate(eval_list):
        if ori_name == target_name:
            target_list.append(target_idx)
            break
print(len(body_joints), len(target_list))
print(target_list)
for idx, name in enumerate(target_list):
    print(idx, eval_list[name])


# print(len(body_joints), len(hand_joints), len(data.joint_names))
# for idx, name in enumerate(data.joint_names):
#     print(idx, name)

# for idx in body_joints:
#     print(data.joint_names[idx])

# print(data._joint_orientation.shape)
# data._joint_orientation = data._joint_orientation[:, 1:, :]
# data._joint_position = data._joint_position[:, 1:, :]
# data._joint_rotation = data._joint_rotation[:, 1:, :]
# data._num_joints = data._num_joints - 1
# data._end_sites = [end_site - 1 for end_site in data._end_sites]
# data._skeleton_joint_offsets = data._skeleton_joint_offsets[1:]
# data._skeleton_joint_parents = [joint_parent - 1 for joint_parent in data._skeleton_joint_parents[1:]]
# data._skeleton_joints = data._skeleton_joints[1:]
# data._joint_translation = data._joint_translation[:, 1:, :]
#
# bvh_save(data, r"F:\jiyi\CAREER_1_4_001_bar_0.bvh")


# print([body_joint - 1 for body_joint in body_joints])
# print([hand_joint - 1 for hand_joint in hand_joints])



# print("new:\n")
# for idx in body:
#     print(data.joint_names[idx])
# for idx in hand:
#     print(data.joint_names[idx])

body = [0, 1, 2, 3, 4, 5, 6, 7, 8, 81, 82, 88, 89, 211, 212, 218, 219, 372, 373, 374, 398, 399, 400]
hand = [91, 92, 93, 113, 114, 115, 134, 135, 136, 152, 153, 154, 174, 175, 176, 221, 222, 223, 243,
                244, 245, 264, 265, 266, 282, 283, 284, 304, 305, 306]

# gt 23+30

# pelvis            0
# spine_01
# spine_02
# spine_03
# spine_04
# spine_05
# neck_01
# neck_02            7    
# head               8
# clavicle_l         81
# upperarm_l         82
# lowerarm_l         88
# hand_l             89
# clavicle_r         211
# upperarm_r         212
# lowerarm_r         218
# hand_r             219
# thigh_r            372
# calf_r             373
# foot_r             374    
# thigh_l            398
# calf_l             399
# foot_l             400
# pinky_01_l
# pinky_02_l
# pinky_03_l
# ring_01_l
# ring_02_l
# ring_03_l
# thumb_01_l
# thumb_02_l
# thumb_03_l
# middle_01_l
# middle_02_l
# middle_03_l
# index_01_l
# index_02_l
# index_03_l
# pinky_01_r
# pinky_02_r
# pinky_03_r
# ring_01_r
# ring_02_r
# ring_03_r
# thumb_01_r
# thumb_02_r
# thumb_03_r
# middle_01_r
# middle_02_r
# middle_03_r
# index_01_r
# index_02_r
# index_03_r

body_smplx = [0, 1, 2, 3, 398, 372, 4, 399, 373, 5, 400, 374, 7, 81, 211, 8, 82, 212, 88, 218, 89, 219, 6]
hand_smplx = [174, 175, 176, 152, 153, 154, 91, 92, 93, 113, 114, 115, 134, 135, 136,
                        304, 305, 306, 282, 283, 284, 221, 222, 223, 243, 244, 245, 264, 265, 266]
# smplx 55
# JOINT_NAMES = [
# 000    "pelvis",
#   1    "left_hip",
#   2   "right_hip", # 
#   3  "spine1",
# 398   "left_knee",
# 372   "right_knee", # 5
#   4  "spine2",
# 399    "left_ankle", # 7
# 373    "right_ankle", # 8
#   5  "spine3",
# 400   "left_foot",
# 374   "right_foot",
#   7   "neck",
#  81   "left_collar",
# 211    "right_collar",
#   8   "head",   # 15
#  82   "left_shoulder",
# 212   "right_shoulder",
#  88   "left_elbow",
# 218   "right_elbow",
#  89   "left_wrist",
# 219   "right_wrist",
#   6  " jaw",    # 22
#     "left_eye_smplhf",
#     "right_eye_smplhf", # 24
#     "left_index1",
#     "left_index2",
#     "left_index3",
#     "left_middle1",
#     "left_middle2",
#     "left_middle3", # 30
#     "left_pinky1",
#     "left_pinky2",
#     "left_pinky3",
#     "left_ring1",
#     "left_ring2",
#     "left_ring3", # 36
#     "left_thumb1",
#     "left_thumb2",
#     "left_thumb3",
#     "right_index1",
#     "right_index2",
#     "right_index3",   # 42
#     "right_middle1",
#     "right_middle2",
#     "right_middle3",
#     "right_pinky1",
#     "right_pinky2",
#     "right_pinky3",   # 48
#     "right_ring1",
#     "right_ring2",
#     "right_ring3",
#     "right_thumb1",
#     "right_thumb2",
#     "right_thumb3",
#     ... ]
