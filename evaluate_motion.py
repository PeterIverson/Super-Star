import os
import glob
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm
import re

# evaluation tools
from emage_evaltools.metric import FGD, BC, L1div

# rotation conversion tools (using the baseline implementation)
from Utils.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    axis_angle_to_rotation_6d,
    euler_angles_to_matrix,
    matrix_to_axis_angle,
)


class BVHParser:
    """Custom BVH file parser"""

    def __init__(self, bvh_path: str):
        self.bvh_path = bvh_path
        self.joint_names = []
        self.joint_offsets = {}
        self.joint_channels = {}
        self.joint_parents = {}
        self.joint_children = {}
        self.channel_order = []
        self.rotation_convention = None  # rotation channel order, e.g. "XYZ" or "ZXY"
        self.num_frames = 0
        self.frame_time = 0.0
        self.motion_data = None

        self._parse_file()

    def _parse_file(self):
        """Parse the BVH file"""
        with open(self.bvh_path, 'r') as f:
            content = f.read()

        parts = content.split('MOTION')
        if len(parts) != 2:
            raise ValueError("Invalid BVH file format: MOTION section not found")

        hierarchy_part = parts[0]
        motion_part = parts[1]

        self._parse_hierarchy(hierarchy_part)
        self._parse_motion(motion_part)

    def _parse_hierarchy(self, hierarchy_str: str):
        """Parse the hierarchy structure"""
        lines = hierarchy_str.strip().split('\n')
        
        joint_stack = []
        current_joint = None
        in_end_site = False
        end_site_depth = 0
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line or line == 'HIERARCHY':
                i += 1
                continue
            
            # handle End Site
            if line.startswith('End Site'):
                in_end_site = True
                end_site_depth = 0
                i += 1
                continue
            
            if in_end_site:
                if line == '{':
                    end_site_depth += 1
                elif line == '}':
                    end_site_depth -= 1
                    if end_site_depth == 0:
                        in_end_site = False
                i += 1
                continue
            
            if line.startswith('ROOT'):
                joint_name = line.split()[1]
                self.joint_names.append(joint_name)
                self.joint_parents[joint_name] = None
                self.joint_children[joint_name] = []
                current_joint = joint_name
                
            elif line.startswith('JOINT'):
                joint_name = line.split()[1]
                self.joint_names.append(joint_name)
                parent = joint_stack[-1] if joint_stack else None
                self.joint_parents[joint_name] = parent
                self.joint_children[joint_name] = []
                if parent:
                    self.joint_children[parent].append(joint_name)
                current_joint = joint_name
                
            elif line.startswith('OFFSET'):
                parts = line.split()
                offset = [float(parts[1]), float(parts[2]), float(parts[3])]
                if current_joint and current_joint in self.joint_names:
                    self.joint_offsets[current_joint] = np.array(offset)
                    
            elif line.startswith('CHANNELS'):
                parts = line.split()
                num_channels = int(parts[1])
                channels = parts[2:2+num_channels]
                if current_joint:
                    self.joint_channels[current_joint] = channels
                    for ch in channels:
                        self.channel_order.append((current_joint, ch))
                    # infer the rotation convention from the rotation channels
                    if self.rotation_convention is None:
                        rot_channels = [ch for ch in channels if 'rotation' in ch]
                        if rot_channels:
                            convention = ''.join([ch[0] for ch in rot_channels])  # e.g. "XYZ" or "ZXY"
                            self.rotation_convention = convention
                        
            elif line == '{':
                if current_joint:
                    joint_stack.append(current_joint)
                    
            elif line == '}':
                if joint_stack:
                    joint_stack.pop()
                    
            i += 1
    
    def _parse_motion(self, motion_str: str):
        """Parse motion data"""
        lines = motion_str.strip().split('\n')
        
        motion_start_idx = 0
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('Frames:'):
                self.num_frames = int(line.split(':')[1].strip())
            elif line.startswith('Frame Time:'):
                self.frame_time = float(line.split(':')[1].strip())
            elif line and not line.startswith('Frames') and not line.startswith('Frame'):
                motion_start_idx = i
                break
        
        motion_list = []
        for line in lines[motion_start_idx:]:
            line = line.strip()
            if line:
                values = [float(x) for x in line.split()]
                motion_list.append(values)
        
        self.motion_data = np.array(motion_list)
        
        # update the actual number of frames
        self.num_frames = len(motion_list)
    
    def get_rotation_data(self) -> np.ndarray:
        """
        Extract rotation data (Euler angles, in degrees)
        The returned data is arranged according to the rotation channel order (convention) in the BVH file.
        For example, when convention="ZXY", rotations[..., 0]=Z rotation, rotations[..., 1]=X rotation, rotations[..., 2]=Y rotation
        so that euler_angles_to_matrix(data, convention) can be used directly in subsequent conversions without reordering.
        """
        num_joints = len(self.joint_names)
        rotations = np.zeros((self.num_frames, num_joints, 3))

        for frame_idx in range(self.num_frames):
            channel_idx = 0
            for joint_idx, joint_name in enumerate(self.joint_names):
                if joint_name not in self.joint_channels:
                    continue

                channels = self.joint_channels[joint_name]
                rot_idx = 0  # index of the rotation component (in channel appearance order)
                
                for ch in channels:
                    if channel_idx >= self.motion_data.shape[1]:
                        break
                    value = self.motion_data[frame_idx, channel_idx]
                    if 'rotation' in ch:
                        if rot_idx < 3:
                            rotations[frame_idx, joint_idx, rot_idx] = value
                            rot_idx += 1
                    channel_idx += 1
                
        return rotations
    
    def get_position_data(self) -> np.ndarray:
        """Extract root joint position data"""
        positions = np.zeros((self.num_frames, 3))
        
        for frame_idx in range(self.num_frames):
            channel_idx = 0
            for joint_name in self.joint_names:
                if joint_name not in self.joint_channels:
                    continue
                    
                channels = self.joint_channels[joint_name]
                
                for ch in channels:
                    if channel_idx >= self.motion_data.shape[1]:
                        break
                    value = self.motion_data[frame_idx, channel_idx]
                    if joint_name == self.joint_names[0]:
                        if ch == 'Xposition':
                            positions[frame_idx, 0] = value
                        elif ch == 'Yposition':
                            positions[frame_idx, 1] = value
                        elif ch == 'Zposition':
                            positions[frame_idx, 2] = value
                    channel_idx += 1
                    
        return positions
    
    def forward_kinematics(self, rotations: np.ndarray = None, root_positions: np.ndarray = None) -> np.ndarray:
        """Forward kinematics to compute world coordinates of all joints
        Args:
            rotations: precomputed rotation data; recomputed internally if None
            root_positions: precomputed root position data; recomputed internally if None
        """
        num_joints = len(self.joint_names)
        positions = np.zeros((self.num_frames, num_joints, 3))
        if rotations is None:
            rotations = self.get_rotation_data()
        if root_positions is None:
            root_positions = self.get_position_data()
        
        # batch convert Euler angles (degrees) to rotation matrices, using the baseline's euler_angles_to_matrix
        convention = self.rotation_convention or 'XYZ'
        euler_rad = np.radians(rotations)  # (T, J, 3)
        euler_tensor = torch.from_numpy(euler_rad).float()
        rot_matrices = euler_angles_to_matrix(euler_tensor, convention).numpy()  # (T, J, 3, 3)
        
        for frame_idx in range(self.num_frames):
            global_transforms = {}
            
            for joint_idx, joint_name in enumerate(self.joint_names):
                local_rot = rot_matrices[frame_idx, joint_idx]  # (3, 3)
                offset = self.joint_offsets.get(joint_name, np.zeros(3))
                
                local_transform = np.eye(4)
                local_transform[:3, :3] = local_rot
                local_transform[:3, 3] = offset
                
                if self.joint_parents[joint_name] is None:
                    local_transform[:3, 3] += root_positions[frame_idx]
                    global_transforms[joint_name] = local_transform
                else:
                    parent_name = self.joint_parents[joint_name]
                    parent_transform = global_transforms[parent_name]
                    global_transforms[joint_name] = parent_transform @ local_transform
                
                positions[frame_idx, joint_idx] = global_transforms[joint_name][:3, 3]
        
        return positions
    
    def get_all_data(self) -> Dict:
        """Get all data"""
        # precompute rotation and position data to avoid redundant computation in forward_kinematics
        rotations = self.get_rotation_data()
        root_positions = self.get_position_data()
        joint_positions = self.forward_kinematics(rotations=rotations, root_positions=root_positions)
        return {
            'joint_names': self.joint_names,
            'num_joints': len(self.joint_names),
            'num_frames': self.num_frames,
            'frame_time': self.frame_time,
            'fps': 1.0 / self.frame_time if self.frame_time > 0 else 30.0,
            'rotation_convention': self.rotation_convention or 'XYZ',
            'rotations': rotations,
            'root_positions': root_positions,
            'joint_positions': joint_positions
        }


class MotionEvaluator:
    """Motion evaluator"""

    def __init__(
        self,
        download_path: str = "./emage_evaltools/",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        sigma: float = 0.3,
        order: int = 7,
        pose_fps: int = 30,
        audio_sr: int = 16000,
        num_joints: int = 55,
        eval_index: Optional[List[int]] = None
    ):
        self.device = device
        self.pose_fps = pose_fps
        self.audio_sr = audio_sr
        self.num_joints = num_joints
        self.download_path = download_path
        self.eval_index = eval_index

        if eval_index is not None:
            print("Evaluation joint list:", eval_index)

        print("Initializing evaluator...")
        self.fgd_evaluator = FGD(download_path=download_path)
        self.bc_evaluator = BC(download_path=download_path, sigma=sigma, order=order)
        self.l1div_evaluator = L1div()

        self.results = []
        self.fgd_pred_list = []
        self.fgd_gt_list = []

    def reset(self):
        """Reset the evaluator"""
        self.fgd_evaluator = FGD(download_path=self.download_path)
        self.bc_evaluator = BC(download_path=self.download_path, sigma=0.3, order=7)
        self.l1div_evaluator = L1div()
        self.results = []
        self.fgd_pred_list = []
        self.fgd_gt_list = []

    def load_bvh(self, bvh_path: str) -> Dict:
        """Load BVH file"""
        parser = BVHParser(bvh_path)
        return parser.get_all_data()
    
    def euler_to_axis_angle_batch(self, euler_deg: np.ndarray, convention: str = "XYZ") -> np.ndarray:
        """
        Batch convert Euler angles (degrees) to axis-angle
        Args:
            euler_deg: (T, J, 3) Euler angles (in degrees) arranged in the convention order
            convention: rotation channel order, e.g. "XYZ", "ZXY", etc.
        """
        T, J, _ = euler_deg.shape
        euler_rad = np.radians(euler_deg)
        euler_tensor = torch.from_numpy(euler_rad).float()
        
        rotation_matrix = euler_angles_to_matrix(euler_tensor, convention)
        axis_angle = matrix_to_axis_angle(rotation_matrix)
        
        return axis_angle.numpy()
    
    def select_joints(
        self, 
        data: np.ndarray, 
        num_original: int,
        target_joints: int = 55,
        axis: int = 1,
        eval_index: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Select/map joints to the target number
        Args:
            data: input data
            num_original: original number of joints
            target_joints: target number of joints
            axis: axis on which the joints are located
            eval_index: list of joint indices to select; if not None, select joints according to this list
        """
        current_joints = data.shape[axis]

        # if eval_index is provided, select joints according to the index list
        if eval_index is not None:
            valid_index = [i for i in eval_index if i < current_joints]
            slices = [slice(None)] * len(data.shape)
            slices[axis] = valid_index
            return data[tuple(slices)]

        if current_joints == target_joints:
            return data
        elif current_joints > target_joints:
            # select the first target_joints joints
            slices = [slice(None)] * len(data.shape)
            slices[axis] = slice(0, target_joints)
            return data[tuple(slices)]
        else:
            # pad with zeros
            pad_width = [(0, 0)] * len(data.shape)
            pad_width[axis] = (0, target_joints - current_joints)
            return np.pad(data, pad_width, mode='constant', constant_values=0)
    
    def evaluate_single(
        self,
        audio_path: str,
        pred_bvh_path: str,
        gt_bvh_path: str,
        skip_start_sec: float = 2.0,
        skip_end_sec: float = 2.0
    ) -> Dict:
        """Evaluate a single sample"""
        # load BVH files
        pred_data = self.load_bvh(pred_bvh_path)
        gt_data = self.load_bvh(gt_bvh_path)

        motion_pred_rot = pred_data['rotations']
        motion_gt_rot = gt_data['rotations']
        motion_pred_pos = pred_data['joint_positions']
        motion_gt_pos = gt_data['joint_positions']

        # pred_data fps is 60, gt_data fps is 30, downsample pred_data to 30fps
        pred_fps = pred_data['fps']
        gt_fps = gt_data['fps']
        if pred_fps > gt_fps and gt_fps > 0:
            downsample_ratio = int(round(pred_fps / gt_fps))
            print(f"  Downsample: pred {pred_fps:.1f}fps -> {gt_fps:.1f}fps (take 1 frame every {downsample_ratio} frames)")
            motion_pred_rot = motion_pred_rot[::downsample_ratio]
            motion_pred_pos = motion_pred_pos[::downsample_ratio]

        # align the number of frames
        t = min(motion_pred_rot.shape[0], motion_gt_rot.shape[0])
        motion_pred_rot = motion_pred_rot[:t]
        motion_gt_rot = motion_gt_rot[:t]
        motion_pred_pos = motion_pred_pos[:t]
        motion_gt_pos = motion_gt_pos[:t]

        actual_fps = gt_fps  # after downsampling, use gt's fps as the actual fps
        num_joints_original = pred_data['num_joints']

        print(f"  Sample info: {t} frames, {num_joints_original} joints, {actual_fps:.1f} FPS (after downsampling)")

        # get their respective rotation conventions
        pred_convention = pred_data['rotation_convention']
        gt_convention = gt_data['rotation_convention']
        print(f"  Rotation convention: pred={pred_convention}, gt={gt_convention}")

        # convert to axis-angle (using their respective rotation conventions)
        motion_pred_aa = self.euler_to_axis_angle_batch(motion_pred_rot, convention=pred_convention)
        motion_gt_aa = self.euler_to_axis_angle_batch(motion_gt_rot, convention=gt_convention)

        # adjust the number of joints
        motion_pred_aa = self.select_joints(motion_pred_aa, num_joints_original, self.num_joints, eval_index=self.eval_index)
        motion_gt_aa = self.select_joints(motion_gt_aa, num_joints_original, self.num_joints, eval_index=self.eval_index)
        motion_pred_pos_adj = self.select_joints(motion_pred_pos, num_joints_original, self.num_joints, eval_index=self.eval_index)
        
        # compute the number of frames to skip
        skip_start_frames = int(skip_start_sec * self.pose_fps)
        skip_end_frames = int(skip_end_sec * self.pose_fps)

        if t <= skip_start_frames + skip_end_frames:
            print(f"  Warning: not enough frames ({t} <= {skip_start_frames + skip_end_frames}), skip this sample")
            return None

        # === BC (Beat Consistency) ===
        if audio_path and os.path.exists(audio_path):
            try:
                # use the adjusted position data
                motion_position_pred = motion_pred_pos_adj.reshape(t, -1)

                audio_beat = self.bc_evaluator.load_audio(
                    audio_path,
                    t_start=int(skip_start_sec * self.audio_sr),
                    t_end=int((t - skip_end_frames) / self.pose_fps * self.audio_sr)
                )
                motion_beat = self.bc_evaluator.load_motion(
                    motion_position_pred,
                    t_start=skip_start_frames,
                    t_end=t - skip_end_frames,
                    pose_fps=self.pose_fps,
                    without_file=True
                )
                self.bc_evaluator.compute(
                    audio_beat,
                    motion_beat,
                    length=t - skip_start_frames - skip_end_frames,
                    pose_fps=self.pose_fps
                )
                print(f"  BC computation done")
            except Exception as e:
                print(f"  BC computation error: {e}")

        # === L1 Diversity ===
        try:
            motion_position_pred = motion_pred_pos_adj.reshape(t, -1)
            self.l1div_evaluator.compute(motion_position_pred)
            print(f"  L1Div computation done")
        except Exception as e:
            print(f"  L1Div computation error: {e}")

        # === FGD (Frechet Gesture Distance) ===
        try:
            motion_gt_tensor = torch.from_numpy(motion_gt_aa).to(self.device).float()
            motion_pred_tensor = torch.from_numpy(motion_pred_aa).to(self.device).float()

            # axis-angle to rotation 6d
            # (T, J, 3) -> (T, J, 6)
            motion_gt_6d = axis_angle_to_rotation_6d(motion_gt_tensor)
            motion_pred_6d = axis_angle_to_rotation_6d(motion_pred_tensor)

            # reshape to (1, T, J*6)
            motion_gt_6d = motion_gt_6d.reshape(1, t, -1)
            motion_pred_6d = motion_pred_6d.reshape(1, t, -1)

            self.fgd_evaluator.update(motion_pred_6d, motion_gt_6d)
            print(f"  FGD update done")
        except Exception as e:
            print(f"  FGD computation error: {e}")
            import traceback
            traceback.print_exc()
        
        return {
            "audio_path": audio_path,
            "pred_bvh_path": pred_bvh_path,
            "gt_bvh_path": gt_bvh_path,
            "num_frames": t,
            "num_joints": num_joints_original,
            "fps": actual_fps
        }
    
    def evaluate_batch(
        self,
        data_list: List[Dict],
        skip_start_sec: float = 2.0,
        skip_end_sec: float = 2.0
    ) -> Dict:
        """Batch evaluation"""
        print(f"Start evaluating {len(data_list)} samples...")

        for idx, data in enumerate(tqdm(data_list, desc="Evaluation progress")):
            print(f"\nProcessing [{idx+1}/{len(data_list)}]: {os.path.basename(data['pred_bvh_path'])}")
            try:
                result = self.evaluate_single(
                    audio_path=data.get("audio_path", ""),
                    pred_bvh_path=data["pred_bvh_path"],
                    gt_bvh_path=data["gt_bvh_path"],
                    skip_start_sec=skip_start_sec,
                    skip_end_sec=skip_end_sec
                )
                if result is not None:
                    self.results.append(result)
            except Exception as e:
                print(f"  Evaluation failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        metrics = self.compute_metrics()
        return metrics

    def compute_metrics(self) -> Dict:
        """Compute all metrics"""
        metrics = {}
        
        try:
            fgd_value = self.fgd_evaluator.compute()
            metrics["fgd"] = float(fgd_value) if fgd_value is not None else None
        except Exception as e:
            print(f"FGD compute error: {e}")
            metrics["fgd"] = None
            
        try:
            bc_value = self.bc_evaluator.avg()
            metrics["bc"] = float(bc_value) if bc_value is not None else None
        except Exception as e:
            print(f"BC compute error: {e}")
            metrics["bc"] = None
            
        try:
            l1_value = self.l1div_evaluator.avg()
            metrics["l1div"] = float(l1_value) if l1_value is not None else None
        except Exception as e:
            print(f"L1Div compute error: {e}")
            metrics["l1div"] = None
            
        metrics["num_samples"] = len(self.results)
        
        return metrics


def find_matching_files(
    audio_dir: str,
    pred_bvh_dir: str,
    gt_bvh_dir: str,
    audio_ext: str = ".wav",
    bvh_ext: str = ".bvh"
) -> List[Dict]:
    """Find matching files"""
    data_list = []

    pred_files = glob.glob(os.path.join(pred_bvh_dir, f"*{bvh_ext}"))

    for pred_path in pred_files:
        base_name = os.path.splitext(os.path.basename(pred_path))[0]

        audio_path = os.path.join(audio_dir, f"{base_name}{audio_ext}") if audio_dir else ""
        gt_bvh_path = os.path.join(gt_bvh_dir, f"{base_name}{bvh_ext}")

        if os.path.exists(gt_bvh_path):
            data_list.append({
                "audio_path": audio_path if audio_path and os.path.exists(audio_path) else "",
                "pred_bvh_path": pred_path,
                "gt_bvh_path": gt_bvh_path
            })
        else:
            print(f"Warning: missing GT file: {gt_bvh_path}")

    print(f"Found {len(data_list)} groups of matching files")
    return data_list


def test_bvh_parser(bvh_path: str):
    """Test the BVH parser"""
    print(f"Test parsing: {bvh_path}")
    parser = BVHParser(bvh_path)
    data = parser.get_all_data()
    
    # eval_index1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 81, 82, 88, 89, 211, 212, 218, 219, 372, 373, 374, 398, 399, 400, 70,22, 
    #               91, 92, 93, 113, 114, 115, 134, 135, 136, 152, 153, 154, 174, 175, 176, 221, 222, 223, 243, 244, 245, 264, 265, 266, 282, 283, 284, 304, 305, 306]
    eval_index = [0, 1, 2, 3, 398, 372, 4, 399, 373, 5, 400, 374, 7, 81, 211, 8, 82, 212, 88, 218, 89, 219, 6, 70,22,
                 174, 175, 176, 152, 153, 154, 91, 92, 93, 113, 114, 115, 134, 135, 136, 304, 305, 306, 282, 283, 284, 221, 222, 223, 243, 244, 245, 264, 265, 266]
    print(f"  Number of joints: {data['num_joints']}")
    joint_names_arr = np.array(data['joint_names'])
    valid_index = [i for i in eval_index if i < len(joint_names_arr)]
    print(f"  Joint names: {joint_names_arr[valid_index].tolist()}...")
    print(f"  Number of frames: {data['num_frames']}")
    print(f"  Frame time: {data['frame_time']:.6f}s")
    print(f"  FPS: {data['fps']:.2f}")
    print(f"  Rotation convention: {data['rotation_convention']}")
    print(f"  Rotation data shape: {data['rotations'].shape}")
    print(f"  Position data shape: {data['joint_positions'].shape}")

    # test rotation conversion
    convention = data['rotation_convention']
    print(f"\nTest rotation conversion (convention={convention})...")
    euler_sample = data['rotations'][:10, :5, :]  # (10, 5, 3)
    euler_rad = np.radians(euler_sample)
    euler_tensor = torch.from_numpy(euler_rad).float()

    rot_matrix = euler_angles_to_matrix(euler_tensor, convention)
    print(f"  Rotation matrix shape: {rot_matrix.shape}")

    axis_angle = matrix_to_axis_angle(rot_matrix)
    print(f"  Axis-angle shape: {axis_angle.shape}")

    rot_6d = axis_angle_to_rotation_6d(axis_angle)
    print(f"  6D rotation shape: {rot_6d.shape}")
    
    return data


def main():
    parser = argparse.ArgumentParser(description="Motion evaluation tool")
    parser.add_argument("--audio_dir", type=str, default="./data_test/audio/",
                        help="audio file directory")
    parser.add_argument("--pred_bvh_dir", type=str, default="./data_test/pred_motion/",
                        help="predicted BVH file directory")
    parser.add_argument("--gt_bvh_dir", type=str, default="./data_test/gt_motion/",
                        help="Ground Truth BVH file directory")
    parser.add_argument("--output_path", type=str, default="./eval_results/eval_results.json",
                        help="result output path")
    parser.add_argument("--download_path", type=str, default="./emage_evaltools/",
                        help="evaluation tool download path")
    parser.add_argument("--skip_start", type=float, default=2.0,
                        help="seconds to skip at the beginning")
    parser.add_argument("--skip_end", type=float, default=2.0,
                        help="seconds to skip at the end")
    parser.add_argument("--pose_fps", type=int, default=30,
                        help="motion frame rate")
    parser.add_argument("--num_joints", type=int, default=55,
                        help="target number of joints")
    parser.add_argument("--eval_index", type=int, nargs='+',
                        default=[0, 1, 2, 3, 398, 372, 4, 399, 373, 5, 400, 374, 7, 81, 211, 8, 82, 212, 88, 218, 89, 219, 6, 70,22,
                 174, 175, 176, 152, 153, 154, 91, 92, 93, 113, 114, 115, 134, 135, 136, 304, 305, 306, 282, 283, 284, 221, 222, 223, 243, 244, 245, 264, 265, 266],
                        help="list of joint indices to evaluate, e.g.: --eval_index 0 1 2 3 4")
    parser.add_argument("--device", type=str, default="cuda",
                        help="compute device")
    parser.add_argument("--test_parser", action="store_true",
                        help="only test the BVH parser")
    
    args = parser.parse_args()
    
    # test mode
    if args.test_parser:
        bvh_files1 = glob.glob(os.path.join(args.pred_bvh_dir, "*.bvh"))
        bvh_files2 = glob.glob(os.path.join(args.gt_bvh_dir, "*.bvh"))
        if bvh_files1:
            test_bvh_parser(bvh_files1[0])
        if bvh_files2:
            test_bvh_parser(bvh_files2[0])
        return

    # check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    # find matching files
    data_list = find_matching_files(
        audio_dir=args.audio_dir,
        pred_bvh_dir=args.pred_bvh_dir,
        gt_bvh_dir=args.gt_bvh_dir
    )

    if len(data_list) == 0:
        print("No matching files found, please check the directory path")
        return

    # initialize the evaluator
    evaluator = MotionEvaluator(
        download_path=args.download_path,
        device=args.device,
        pose_fps=args.pose_fps,
        num_joints=args.num_joints,
        eval_index=args.eval_index
    )

    # batch evaluation
    metrics = evaluator.evaluate_batch(
        data_list=data_list,
        skip_start_sec=args.skip_start,
        skip_end_sec=args.skip_end
    )

    # print results
    print("\n" + "=" * 50)
    print("Evaluation results:")
    print("=" * 50)
    for key, value in metrics.items():
        if value is not None:
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        else:
            print(f"  {key}: N/A")
    print("=" * 50)

    # save results
    # e.g.: pred_bvh_dir = "./data/pred_motion_layer1/" -> suffix is "layer1"
    pred_dir = args.pred_bvh_dir.rstrip("/\\")
    base = os.path.basename(pred_dir)                 # "pred_motion_layer1"
    suffix = base.split("pred_motion_", 1)[-1]        # "layer1" (or the original base if not contained)

    args.output_path = f"./eval_results/eval_results_{suffix}.json"

    import json
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {args.output_path}")


if __name__ == "__main__":
    main()
