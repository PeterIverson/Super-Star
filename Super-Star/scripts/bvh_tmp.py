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

data = bvh_load(r"F:\jiyi\20250811\CAREER_1_4_001_bar_0.bvh")
print(data.joint_names, len(data.joint_names))
body_joints = [1, 2, 3, 4, 5, 6, 7, 8, 9, 106, 107, 117, 118, 327, 328, 338, 339, 607, 608, 609, 661, 662, 663]
hand_joints = [120, 121, 122, 157, 158, 159, 193, 194, 195, 223, 224, 225, 260, 261, 262, 341, 342, 343,
               378, 379, 380, 414, 415, 416, 444, 445, 446, 481, 482, 483]
data_new = bvh_load(r"F:\jiyi\20250818\CAREER_1_4_001_bar_0.bvh")
print(data_new.joint_names, len(data_new.joint_names))

target_list = []
for body_idx in body_joints:
    ori_name = data.joint_names[body_idx].replace("DHIbody:", "")
    for target_idx, target_name in enumerate(data_new.joint_names):
        if ori_name == target_name:
            target_list.append(target_idx)
            break
print(len(body_joints), len(target_list))
print(target_list)


# print(len(body_joints), len(hand_joints), len(data.joint_names))
# # for idx, name in enumerate(data.joint_names):
# #     print(idx, name)
#
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
body = [0, 1, 2, 3, 4, 5, 6, 7, 8, 93, 94, 104, 105, 313, 314, 324, 325, 592, 593, 594, 638, 639, 640]
hand = [107, 108, 109, 144, 145, 146, 180, 181, 182, 210, 211, 212, 247, 248, 249, 327,
        328, 329, 364, 365, 366, 400, 401, 402, 430, 431, 432, 467, 468, 469]
