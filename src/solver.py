import numpy as np
from scipy.spatial.transform import Rotation as R

class PoseSolver:
    def __init__(self):
        # Define the standard T-Pose vectors for a human skeleton in Blender's coordinate system (Z-up, Y-forward)
        # This assumes the character is facing -Y or +Y? Usually characters face -Y in Blender (Front Orthographic).
        # Let's assume standard Rigify/Mixamo:
        # T-Pose: Arms along X axis, Legs along -Z axis, Spine along +Z axis.
        
        self.rest_vectors = {
            "spine": np.array([0, 0, 1]),      # Up
            "left_arm": np.array([1, 0, 0]),   # Out to Left
            "left_forearm": np.array([1, 0, 0]),
            "right_arm": np.array([-1, 0, 0]), # Out to Right
            "right_forearm": np.array([-1, 0, 0]),
            "left_up_leg": np.array([0, 0, -1]), # Down
            "left_leg": np.array([0, 0, -1]),
            "right_up_leg": np.array([0, 0, -1]),
            "right_leg": np.array([0, 0, -1]),
            "neck": np.array([0, 0, 1]),
        }
        self.initial_hip_pos = None

    def calibrate(self, landmarks):
        """
        Captures the current pose as the Rest Pose (T-Pose).
        Updates self.rest_vectors with the vectors from the current landmarks.
        """
        print("Calibrating... Please stand in T-Pose.")
        
        # Convert MP to Blender coords for calibration
        converted_lm = []
        for lm in landmarks:
            class Point:
                def __init__(self, x, y, z):
                    self.x = x
                    # DAMPEN DEPTH (Consistent with solve)
                    self.y = z * 0.5
                    self.z = -y
            converted_lm.append(Point(lm.x, lm.y, lm.z))

        # Helper to get vector
        def get_calib_vec(p_start_idx, p_end_idx):
            start = converted_lm[p_start_idx]
            end = converted_lm[p_end_idx]
            return self.get_vector(start, end)

        # Update Rest Vectors based on USER'S ACTUAL BODY
        # Spine
        mid_hip = type('P', (), {})()
        mid_hip.x = (converted_lm[23].x + converted_lm[24].x) / 2
        mid_hip.y = (converted_lm[23].y + converted_lm[24].y) / 2
        mid_hip.z = (converted_lm[23].z + converted_lm[24].z) / 2
        
        mid_shoulder = type('P', (), {})()
        mid_shoulder.x = (converted_lm[11].x + converted_lm[12].x) / 2
        mid_shoulder.y = (converted_lm[11].y + converted_lm[12].y) / 2
        mid_shoulder.z = (converted_lm[11].z + converted_lm[12].z) / 2
        
        self.rest_vectors["spine"] = self.get_vector(mid_hip, mid_shoulder)
        
        # Neck (Mid-shoulder to Nose)
        if len(converted_lm) > 0:
            nose = converted_lm[0]
            self.rest_vectors["neck"] = self.get_vector(mid_shoulder, nose)

        # Arms
        self.rest_vectors["left_arm"] = get_calib_vec(11, 13)
        self.rest_vectors["left_forearm"] = get_calib_vec(13, 15)
        self.rest_vectors["right_arm"] = get_calib_vec(12, 14)
        self.rest_vectors["right_forearm"] = get_calib_vec(14, 16)
        
        # Legs
        self.rest_vectors["left_up_leg"] = get_calib_vec(23, 25)
        self.rest_vectors["left_leg"] = get_calib_vec(25, 27)
        self.rest_vectors["right_up_leg"] = get_calib_vec(24, 26)
        self.rest_vectors["right_leg"] = get_calib_vec(26, 28)
        
        # Reset Hip Position
        bx = mid_hip.x
        by = mid_hip.y
        bz = mid_hip.z
        self.initial_hip_pos = np.array([bx, by, bz])
        
        print("Calibration Complete.")

    def calculate_root_translation(self, landmarks):
        """
        Calculate the translation of the root (hips) relative to the initial position.
        Returns [x, y, z] vector in Blender units.
        """
        # Hip Center in MediaPipe
        # 23: Left Hip, 24: Right Hip
        # MP Coords: X (0-1), Y (0-1), Z (approx scale)
        
        # We need to convert to Blender coords first:
        # Blender X = MP X
        # Blender Y = MP Z
        # Blender Z = -MP Y
        
        # Get raw MP coords for hips
        l_hip = landmarks[23]
        r_hip = landmarks[24]
        
        # Average to get center
        cx = (l_hip.x + r_hip.x) / 2
        cy = (l_hip.y + r_hip.y) / 2
        cz = (l_hip.z + r_hip.z) / 2
        
        # Convert to Blender Coords (Raw, normalized 0-1)
        # Note: Z in MP is not normalized 0-1, it's relative to image width.
        bx = cx
        # DAMPEN DEPTH (Consistent with solve/calibrate)
        by = cz * 0.5
        bz = -cy
        
        current_pos = np.array([bx, by, bz])
        
        if self.initial_hip_pos is None:
            self.initial_hip_pos = current_pos
            return [0.0, 0.0, 0.0]
            
        # Calculate Delta
        delta = current_pos - self.initial_hip_pos
        
        # Scale Factor
        # MediaPipe coordinates are small. We need to scale them up for Blender.
        # A human is ~1.7m tall. In MP, height is ~0.5-0.8 units?
        # Let's try a scale factor of 2.0 or 3.0 initially.
        SCALE = 3.0
        
        # Invert X because MP is mirrored?
        # Usually webcam is mirrored. If user moves left, image moves right.
        # Blender X+ is Right.
        # Let's keep X as is for now, might need -X.
        
        translation = delta * SCALE
        
        # We might want to zero out Y (Forward/Back) if depth is noisy
        # translation[1] = 0 
        
        return translation.tolist()

    def get_vector(self, p1, p2):
        """Calculate vector from p1 to p2."""
        return np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm

    def rotation_between_vectors(self, u, v):
        """
        Calculate quaternion rotation that aligns vector u to vector v.
        Formula: q = cos(theta/2) + sin(theta/2) * rotation_axis
        """
        u = self.normalize(u)
        v = self.normalize(v)

        if np.allclose(u, v):
            return np.array([1, 0, 0, 0]) # w, x, y, z
        
        if np.allclose(u, -v):
            # 180 degree rotation. Axis can be any vector orthogonal to u.
            # This is a rare edge case in mocap (bones don't flip 180 instantly).
            return np.array([0, 1, 0, 0])

        # Cross product gives the axis of rotation
        axis = np.cross(u, v)
        
        # Dot product gives cosine of angle
        dot = np.dot(u, v)
        
        # Calculate angle
        # dot = cos(theta)
        # We need half angle for quaternion
        # A stable way to compute the quaternion is:
        # q = [sqrt(2(1+dot)), axis_x, axis_y, axis_z] normalized? 
        # Or standard:
        # theta = arccos(dot)
        # axis = axis / sin(theta)
        
        # Using scipy for robustness, but explaining the math:
        # We want R such that R * u = v
        
        # Manual implementation as requested:
        # q_w = sqrt(|u|^2 * |v|^2) + dot(u, v)
        # q_xyz = cross(u, v)
        
        q_xyz = axis
        q_w = np.sqrt(np.linalg.norm(u)**2 * np.linalg.norm(v)**2) + dot
        
        # Normalize quaternion
        q = np.array([q_w, q_xyz[0], q_xyz[1], q_xyz[2]])
        return self.normalize(q)

    def multiply_quaternions(self, q1, q2):
        """
        Multiply two quaternions q1 * q2.
        q = [w, x, y, z]
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return np.array([w, x, y, z])

    def invert_quaternion(self, q):
        """
        Invert a quaternion.
        q_inv = [w, -x, -y, -z] / norm^2
        Assumes unit quaternion, so just conjugate.
        """
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def solve(self, landmarks):
        """
        Input: MediaPipe landmarks (list of objects with x, y, z, visibility)
        Output: Dictionary of bone names to [w, x, y, z]
        """
        
        # 1. Convert MediaPipe coordinates to Blender coordinates
        converted_lm = []
        for lm in landmarks:
            class Point:
                def __init__(self, x, y, z):
                    # MediaPipe: X(0-1 left-right), Y(0-1 top-bottom), Z(depth)
                    # Blender: X(left-right), Y(back-front), Z(down-up)
                    # Conversion:
                    # - X as-is (no inversion needed, will handle mirror in IK)
                    # - Swap Y and Z axes
                    # - Invert Y (was Z in MP) for Z-up
                    self.x = x      # No inversion
                    self.y = z      # Depth becomes Y (forward/back)
                    self.z = -y     # Height becomes Z (up/down)
            converted_lm.append(Point(lm.x, lm.y, lm.z))

        bones = {}
        
        # Helper to get GLOBAL rotation for a specific bone
        def calc_global_rot(p_start_idx, p_end_idx, rest_vec_name):
            start = converted_lm[p_start_idx]
            end = converted_lm[p_end_idx]
            current_vec = self.get_vector(start, end)
            rest_vec = self.rest_vectors[rest_vec_name]
            return self.rotation_between_vectors(rest_vec, current_vec)

        # --- 1. Calculate GLOBAL Rotations first ---
        
        # Spine / Torso
        mid_hip = type('P', (), {})()
        mid_hip.x = (converted_lm[23].x + converted_lm[24].x) / 2
        mid_hip.y = (converted_lm[23].y + converted_lm[24].y) / 2
        mid_hip.z = (converted_lm[23].z + converted_lm[24].z) / 2
        
        mid_shoulder = type('P', (), {})()
        mid_shoulder.x = (converted_lm[11].x + converted_lm[12].x) / 2
        mid_shoulder.y = (converted_lm[11].y + converted_lm[12].y) / 2
        mid_shoulder.z = (converted_lm[11].z + converted_lm[12].z) / 2
        
        spine_vec = self.get_vector(mid_hip, mid_shoulder)
        q_spine_global = self.rotation_between_vectors(self.rest_vectors["spine"], spine_vec)
        bones["spine"] = q_spine_global.tolist()

        # Neck
        q_neck_global = np.array([1, 0, 0, 0])
        if len(converted_lm) > 0:
            nose = converted_lm[0]
            neck_vec = self.get_vector(mid_shoulder, nose)
            q_neck_global = self.rotation_between_vectors(self.rest_vectors["neck"], neck_vec)
            
            q_spine_inv = self.invert_quaternion(q_spine_global)
            q_neck_local = self.multiply_quaternions(q_spine_inv, q_neck_global)
            bones["neck"] = q_neck_local.tolist()
            
            # Head (Identity for now)
            bones["head"] = [1, 0, 0, 0]

        # --- Arms ---
        
        # Left Arm
        q_upper_arm_L_global = calc_global_rot(11, 13, "left_arm")
        q_forearm_L_global = calc_global_rot(13, 15, "left_forearm")
        
        q_upper_arm_L_local = self.multiply_quaternions(self.invert_quaternion(q_spine_global), q_upper_arm_L_global)
        bones["upper_arm.L"] = q_upper_arm_L_local.tolist()
        
        q_upper_arm_L_inv = self.invert_quaternion(q_upper_arm_L_global)
        q_forearm_L_local = self.multiply_quaternions(q_upper_arm_L_inv, q_forearm_L_global)
        bones["forearm.L"] = q_forearm_L_local.tolist()
        
        # Right Arm
        q_upper_arm_R_global = calc_global_rot(12, 14, "right_arm")
        q_forearm_R_global = calc_global_rot(14, 16, "right_forearm")
        
        q_upper_arm_R_local = self.multiply_quaternions(self.invert_quaternion(q_spine_global), q_upper_arm_R_global)
        bones["upper_arm.R"] = q_upper_arm_R_local.tolist()
        
        q_upper_arm_R_inv = self.invert_quaternion(q_upper_arm_R_global)
        q_forearm_R_local = self.multiply_quaternions(q_upper_arm_R_inv, q_forearm_R_global)
        bones["forearm.R"] = q_forearm_R_local.tolist()
        
        # --- Legs ---
        
        # Left Leg
        q_thigh_L_global = calc_global_rot(23, 25, "left_up_leg")
        q_shin_L_global = calc_global_rot(25, 27, "left_leg")
        
        q_thigh_L_local = self.multiply_quaternions(self.invert_quaternion(q_spine_global), q_thigh_L_global)
        bones["thigh.L"] = q_thigh_L_local.tolist()
        
        q_thigh_L_inv = self.invert_quaternion(q_thigh_L_global)
        q_shin_L_local = self.multiply_quaternions(q_thigh_L_inv, q_shin_L_global)
        bones["shin.L"] = q_shin_L_local.tolist()
        
        # Right Leg
        q_thigh_R_global = calc_global_rot(24, 26, "right_up_leg")
        q_shin_R_global = calc_global_rot(26, 28, "right_leg")
        
        q_thigh_R_local = self.multiply_quaternions(self.invert_quaternion(q_spine_global), q_thigh_R_global)
        bones["thigh.R"] = q_thigh_R_local.tolist()
        
        q_thigh_R_inv = self.invert_quaternion(q_thigh_R_global)
        q_shin_R_local = self.multiply_quaternions(q_thigh_R_inv, q_shin_R_global)
        bones["shin.R"] = q_shin_R_local.tolist()

        # --- IK Targets: Hands and Feet (Positions) ---
        # For IK, we need to send POSITIONS in world space, relative to the hip center
        
        # Get hip center as reference
        return translation.tolist()

    def get_vector(self, p1, p2):
        """Calculate vector from p1 to p2."""
        return np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm

    def rotation_between_vectors(self, u, v):
        """
        Calculate quaternion rotation that aligns vector u to vector v.
        Formula: q = cos(theta/2) + sin(theta/2) * rotation_axis
        """
        u = self.normalize(u)
        v = self.normalize(v)

        if np.allclose(u, v):
            return np.array([1, 0, 0, 0]) # w, x, y, z
        
        if np.allclose(u, -v):
            # 180 degree rotation. Axis can be any vector orthogonal to u.
            # This is a rare edge case in mocap (bones don't flip 180 instantly).
            return np.array([0, 1, 0, 0])

        # Cross product gives the axis of rotation
        axis = np.cross(u, v)
        
        # Dot product gives cosine of angle
        dot = np.dot(u, v)
        
        # Calculate angle
        # dot = cos(theta)
        # We need half angle for quaternion
        # A stable way to compute the quaternion is:
        # q = [sqrt(2(1+dot)), axis_x, axis_y, axis_z] normalized? 
        # Or standard:
        # theta = arccos(dot)
        # axis = axis / sin(theta)
        
        # Using scipy for robustness, but explaining the math:
        # We want R such that R * u = v
        
        # Manual implementation as requested:
        # q_w = sqrt(|u|^2 * |v|^2) + dot(u, v)
        # q_xyz = cross(u, v)
        
        q_xyz = axis
        q_w = np.sqrt(np.linalg.norm(u)**2 * np.linalg.norm(v)**2) + dot
        
        # Normalize quaternion
        q = np.array([q_w, q_xyz[0], q_xyz[1], q_xyz[2]])
        return self.normalize(q)

    def multiply_quaternions(self, q1, q2):
        """
        Multiply two quaternions q1 * q2.
        q = [w, x, y, z]
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return np.array([w, x, y, z])

    def invert_quaternion(self, q):
        """
        Invert a quaternion.
        q_inv = [w, -x, -y, -z] / norm^2
        Assumes unit quaternion, so just conjugate.
        """
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def scale_quaternion_rotation(self, q, factor):
        """
        Scale the rotation angle of a quaternion by a factor.
        q = [w, x, y, z]
        """
        # Extract angle and axis
        # w = cos(theta/2)
        # xyz = sin(theta/2) * axis
        
        w = q[0]
        # Clamp w to [-1, 1] to avoid numerical errors
        w = max(-1.0, min(1.0, w))
        
        theta = 2 * np.arccos(w)
        
        if abs(np.sin(theta/2)) < 1e-6:
            return q # Angle is 0, no scaling needed
            
        axis = q[1:] / np.sin(theta/2)
        
        # Scale angle
        new_theta = theta * factor
        
        # Reconstruct quaternion
        new_w = np.cos(new_theta/2)
        new_xyz = np.sin(new_theta/2) * axis
        
        return np.array([new_w, new_xyz[0], new_xyz[1], new_xyz[2]])

    def solve(self, results):
        """
        Input: MediaPipe Holistic results object (contains pose_landmarks, left_hand_landmarks, right_hand_landmarks)
        Output: Dictionary of bone names to [w, x, y, z] (Rotations) or [x, y, z] (Positions)
        """
        
        if not results.pose_landmarks:
            return {}

        # Helper class for coordinate conversion
        class Point:
            def __init__(self, x, y, z):
                # MediaPipe: X(0-1 left-right), Y(0-1 top-bottom), Z(depth)
                # Blender: X(left-right), Y(back-front), Z(down-up)
                self.x = x      # No inversion
                # DAMPEN DEPTH (Z in MP -> Y in Blender)
                # We multiply by 0.5 to reduce the "leaning" effect caused by exaggerated depth estimation.
                self.y = z * 0.5 
                self.z = -y     # Height becomes Z (up/down)

        # 1. Convert POSE landmarks
        pose_lm = []
        for lm in results.pose_landmarks.landmark:
            pose_lm.append(Point(lm.x, lm.y, lm.z))

        # 2. Convert HAND landmarks (if available)
        left_hand_lm = []
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                left_hand_lm.append(Point(lm.x, lm.y, lm.z))
                
        right_hand_lm = []
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                right_hand_lm.append(Point(lm.x, lm.y, lm.z))

        bones = {}
        
        # Helper to get GLOBAL rotation for a specific bone (Generic)
        def calc_global_rot(landmarks_list, p_start_idx, p_end_idx, rest_vec_name):
            if not landmarks_list or p_start_idx >= len(landmarks_list) or p_end_idx >= len(landmarks_list):
                return np.array([1, 0, 0, 0])
                
            start = landmarks_list[p_start_idx]
            end = landmarks_list[p_end_idx]
            current_vec = self.get_vector(start, end)
            
            if rest_vec_name in self.rest_vectors:
                rest_vec = self.rest_vectors[rest_vec_name]
            else:
                # Fallback defaults
                if "left" in rest_vec_name: rest_vec = np.array([1, 0, 0])
                elif "right" in rest_vec_name: rest_vec = np.array([-1, 0, 0])
                else: rest_vec = np.array([0, 0, 1])
                
            return self.rotation_between_vectors(rest_vec, current_vec)

        # --- 1. SPINE & HEAD (FK) ---
        # Uses POSE landmarks
        
        mid_hip = type('P', (), {})()
        mid_hip.x = (pose_lm[23].x + pose_lm[24].x) / 2
        mid_hip.y = (pose_lm[23].y + pose_lm[24].y) / 2
        mid_hip.z = (pose_lm[23].z + pose_lm[24].z) / 2
        
        mid_shoulder = type('P', (), {})()
        mid_shoulder.x = (pose_lm[11].x + pose_lm[12].x) / 2
        mid_shoulder.y = (pose_lm[11].y + pose_lm[12].y) / 2
        mid_shoulder.z = (pose_lm[11].z + pose_lm[12].z) / 2
        
        # --- TORSO (Main Body Rotation) ---
        # Vector from Hips to Shoulders
        spine_vec = self.get_vector(mid_hip, mid_shoulder)
        
        # Calculate Global Torso Rotation
        # Rest Vector for Spine/Torso is +Z (Up)
        q_torso_global = self.rotation_between_vectors(np.array([0, 0, 1]), spine_vec)
        
        # Apply to "torso" bone (Root of the spine chain)
        bones["torso_rot"] = q_torso_global.tolist()
        
        # --- SPINE CURVATURE ---
        # Apply a fraction of the torso rotation to the spine bones to create a curve.
        # Since spine_fk is a child of torso, this adds to the existing lean.
        q_spine_curve = self.scale_quaternion_rotation(q_torso_global, 0.2) # 20% curve
        
        bones["spine_fk"] = q_spine_curve.tolist()
        bones["spine_fk.001"] = q_spine_curve.tolist()

        # --- HEAD ---
        # Calculate Head Rotation
        
        if len(pose_lm) > 0:
            # Construct a "Head Up" vector
            # Mid-Shoulder (Neck Base) -> Mid-Ear (Head Center)
            l_ear = pose_lm[7]
            r_ear = pose_lm[8]
            
            mid_ear = type('P', (), {})()
            mid_ear.x = (l_ear.x + r_ear.x) / 2
            mid_ear.y = (l_ear.y + r_ear.y) / 2
            mid_ear.z = (l_ear.z + r_ear.z) / 2
            
            head_vec = self.get_vector(mid_shoulder, mid_ear)
            
            # Calculate Global Head Rotation
            # Rest vector for Head is [0, 0, 1] (Up)
            q_head_global = self.rotation_between_vectors(np.array([0, 0, 1]), head_vec)
            
            # Apply to Head bone
            # Head is child of Neck -> Spine... -> Torso.
            # We need Head Local.
            # Parent Global is roughly Torso Global (since Spine/Neck are small).
            # To be precise, Parent Global = Torso * Spine_Curve * Spine_Curve...
            # But approximating with Torso Global is usually fine for Head stability.
            q_parent_global = q_torso_global
            
            q_head_local = self.multiply_quaternions(self.invert_quaternion(q_parent_global), q_head_global)
            
            # EXAGGERATE HEAD MOVEMENT
            q_head_local = self.scale_quaternion_rotation(q_head_local, 1.5) # 150% movement
            
            bones["head_fk"] = q_head_local.tolist()
            bones["neck"] = [1, 0, 0, 0]
        else:
            bones["head_fk"] = [1, 0, 0, 0]
            bones["neck"] = [1, 0, 0, 0]
            q_head_global = q_torso_global # Fallback

        # --- 2. ARMS (FK) ---
        # MIRROR MODE: User Left -> Character Right, User Right -> Character Left
        
        # We need to use q_torso_global as the parent rotation for the arms
        # because the arms are attached to the torso (via shoulder/clavicle).
        q_spine_global = q_torso_global # Alias for compatibility with existing arm logic
        
        # Left Arm (Character) <- Right Arm (User)
        # Landmarks: 12 (R Shoulder), 14 (R Elbow), 16 (R Wrist)
        q_upper_arm_L_global = calc_global_rot(pose_lm, 12, 14, "right_arm")
        q_forearm_L_global = calc_global_rot(pose_lm, 14, 16, "right_forearm")
        
        q_upper_arm_L_local = self.multiply_quaternions(self.invert_quaternion(q_spine_global), q_upper_arm_L_global)
        bones["upper_arm_fk.L"] = q_upper_arm_L_local.tolist()
        
        q_upper_arm_L_inv = self.invert_quaternion(q_upper_arm_L_global)
        q_forearm_L_local = self.multiply_quaternions(q_upper_arm_L_inv, q_forearm_L_global)
        bones["forearm_fk.L"] = q_forearm_L_local.tolist()
        
        # Right Arm (Character) <- Left Arm (User)
        # Landmarks: 11 (L Shoulder), 13 (L Elbow), 15 (L Wrist)
        q_upper_arm_R_global = calc_global_rot(pose_lm, 11, 13, "left_arm")
        q_forearm_R_global = calc_global_rot(pose_lm, 13, 15, "left_forearm")
        
        q_upper_arm_R_local = self.multiply_quaternions(self.invert_quaternion(q_spine_global), q_upper_arm_R_global)
        bones["upper_arm_fk.R"] = q_upper_arm_R_local.tolist()
        
        q_upper_arm_R_inv = self.invert_quaternion(q_upper_arm_R_global)
        q_forearm_R_local = self.multiply_quaternions(q_upper_arm_R_inv, q_forearm_R_global)
        bones["forearm_fk.R"] = q_forearm_R_local.tolist()

        # --- 3. LEGS (FK) ---
        # MIRROR MODE
        
        # Left Leg (Character) <- Right Leg (User)
        # Landmarks: 24 (R Hip), 26 (R Knee), 28 (R Ankle)
        q_thigh_L_global = calc_global_rot(pose_lm, 24, 26, "right_up_leg")
        q_shin_L_global = calc_global_rot(pose_lm, 26, 28, "right_leg")
        
        q_thigh_L_local = self.multiply_quaternions(self.invert_quaternion(q_spine_global), q_thigh_L_global)
        bones["thigh_fk.L"] = q_thigh_L_local.tolist()
        
        q_thigh_L_inv = self.invert_quaternion(q_thigh_L_global)
        q_shin_L_local = self.multiply_quaternions(q_thigh_L_inv, q_shin_L_global)
        bones["shin_fk.L"] = q_shin_L_local.tolist()
        
        # Right Leg (Character) <- Left Leg (User)
        # Landmarks: 23 (L Hip), 25 (L Knee), 27 (L Ankle)
        q_thigh_R_global = calc_global_rot(pose_lm, 23, 25, "left_up_leg")
        q_shin_R_global = calc_global_rot(pose_lm, 25, 27, "left_leg")
        
        q_thigh_R_local = self.multiply_quaternions(self.invert_quaternion(q_spine_global), q_thigh_R_global)
        bones["thigh_fk.R"] = q_thigh_R_local.tolist()
        
        q_thigh_R_inv = self.invert_quaternion(q_thigh_R_global)
        q_shin_R_local = self.multiply_quaternions(q_thigh_R_inv, q_shin_R_global)
        bones["shin_fk.R"] = q_shin_R_local.tolist()

        # --- 4. FINGERS (FK) ---
        # MIRROR MODE
        
        def solve_finger_chain(lm_list, prefix, indices, parent_global_rot, side_name):
            if not lm_list: return
            
            # Note: side_name passed here is the CHARACTER side (e.g., "left")
            # But we need to use the REST VECTOR of the USER side? 
            # No, we are mapping User Right Hand -> Character Left Hand.
            # The "rest_vec_name" in calc_global_rot uses "left_arm" etc.
            # If we are solving for Character Left Hand, we should use "left_arm" rest vector?
            # Actually, the landmarks are from User Right Hand. 
            # User Right Hand rest vector is "right_arm".
            # So we should use "right_arm" rest vector for calculating rotation of User Right Hand.
            # And then apply that rotation to Character Left Hand.
            
            # Determine source side for rest vector lookup
            source_side = "right" if side_name == "left" else "left"
            
            # 1. Proximal (MCP -> PIP)
            q_prox_global = calc_global_rot(lm_list, indices[0], indices[1], f"{source_side}_arm") 
            q_prox_local = self.multiply_quaternions(self.invert_quaternion(parent_global_rot), q_prox_global)
            bones[f"{prefix}.01.{side_name[0].upper()}"] = q_prox_local.tolist()
            
            # 2. Intermediate (PIP -> DIP)
            q_inter_global = calc_global_rot(lm_list, indices[1], indices[2], f"{source_side}_arm")
            q_inter_local = self.multiply_quaternions(self.invert_quaternion(q_prox_global), q_inter_global)
            bones[f"{prefix}.02.{side_name[0].upper()}"] = q_inter_local.tolist()
            
            # 3. Distal (DIP -> Tip)
            q_dist_global = calc_global_rot(lm_list, indices[2], indices[3], f"{source_side}_arm")
            q_dist_local = self.multiply_quaternions(self.invert_quaternion(q_inter_global), q_dist_global)
            bones[f"{prefix}.03.{side_name[0].upper()}"] = q_dist_local.tolist()

        # Left Hand (Character) <- Right Hand (User)
        if right_hand_lm:
            # Calculate Hand Global Rotation (Wrist -> Middle MCP)
            q_hand_L_global = calc_global_rot(right_hand_lm, 0, 9, "right_forearm")
            bones["hand_fk.L"] = self.multiply_quaternions(self.invert_quaternion(q_forearm_L_global), q_hand_L_global).tolist()
            
            # Fingers
            solve_finger_chain(right_hand_lm, "thumb", [1, 2, 3, 4], q_hand_L_global, "left")
            solve_finger_chain(right_hand_lm, "f_index", [5, 6, 7, 8], q_hand_L_global, "left")
            solve_finger_chain(right_hand_lm, "f_middle", [9, 10, 11, 12], q_hand_L_global, "left")
            solve_finger_chain(right_hand_lm, "f_ring", [13, 14, 15, 16], q_hand_L_global, "left")
            solve_finger_chain(right_hand_lm, "f_pinky", [17, 18, 19, 20], q_hand_L_global, "left")
            
        # Right Hand (Character) <- Left Hand (User)
        if left_hand_lm:
            # Calculate Hand Global Rotation (Wrist -> Middle MCP)
            q_hand_R_global = calc_global_rot(left_hand_lm, 0, 9, "left_forearm")
            bones["hand_fk.R"] = self.multiply_quaternions(self.invert_quaternion(q_forearm_R_global), q_hand_R_global).tolist()
            
            # Fingers
            solve_finger_chain(left_hand_lm, "thumb", [1, 2, 3, 4], q_hand_R_global, "right")
            solve_finger_chain(left_hand_lm, "f_index", [5, 6, 7, 8], q_hand_R_global, "right")
            solve_finger_chain(left_hand_lm, "f_middle", [9, 10, 11, 12], q_hand_R_global, "right")
            solve_finger_chain(left_hand_lm, "f_ring", [13, 14, 15, 16], q_hand_R_global, "right")
            solve_finger_chain(left_hand_lm, "f_pinky", [17, 18, 19, 20], q_hand_R_global, "right")


        # --- 5. IK TARGETS (Hybrid Mode) ---
        # MIRROR MODE
        
        # Root / Hips Translation
        bones["hips"] = self.calculate_root_translation(results.pose_landmarks.landmark)
        
        # Get hip center as reference
        mid_hip_pos = np.array([mid_hip.x, mid_hip.y, mid_hip.z])
        
        # Scale factor (calibrated for character size ~7 units tall)
        SCALE = 14.0
        
        # mid_hip_raw for coordinate conversion consistency (Average of Left and Right Hip)
        l_hip = results.pose_landmarks.landmark[23]
        r_hip = results.pose_landmarks.landmark[24]
        mid_hip_raw = np.array([
            -(l_hip.x + r_hip.x) / 2, 
            ((l_hip.z + r_hip.z) / 2) * 0.5, # DAMPEN DEPTH
            -(l_hip.y + r_hip.y) / 2
        ])
        
        # 1. Hands (Wrist Position)
        # Left Hand (Character) <- Right Wrist (User)
        l_wrist = results.pose_landmarks.landmark[16] # 16 is Right Wrist
        l_hand_raw = np.array([-l_wrist.x, l_wrist.z * 0.5, -l_wrist.y])
        l_hand_offset = (l_hand_raw - mid_hip_raw) * SCALE
        bones["hand_ik.L"] = l_hand_offset.tolist()
        
        # Right Hand (Character) <- Left Wrist (User)
        r_wrist = results.pose_landmarks.landmark[15] # 15 is Left Wrist
        r_hand_raw = np.array([-r_wrist.x, r_wrist.z * 0.5, -r_wrist.y])
        r_hand_offset = (r_hand_raw - mid_hip_raw) * SCALE
        bones["hand_ik.R"] = r_hand_offset.tolist()
        
        # 2. Feet (Ankle Position)
        # Left Foot (Character) <- Right Ankle (User)
        l_ankle = results.pose_landmarks.landmark[28] # 28 is Right Ankle
        l_foot_raw = np.array([-l_ankle.x, l_ankle.z * 0.5, -l_ankle.y])
        l_foot_offset = (l_foot_raw - mid_hip_raw) * SCALE
        bones["foot_ik.L"] = l_foot_offset.tolist()
        
        # Right Foot (Character) <- Left Ankle (User)
        r_ankle = results.pose_landmarks.landmark[27] # 27 is Left Ankle
        r_foot_raw = np.array([-r_ankle.x, r_ankle.z * 0.5, -r_ankle.y])
        r_foot_offset = (r_foot_raw - mid_hip_raw) * SCALE
        bones["foot_ik.R"] = r_foot_offset.tolist()
        
        # 3. Pole Vectors (Elbows and Knees)
        # Left Elbow Target (Character) <- Right Elbow (User)
        l_elbow = results.pose_landmarks.landmark[14] # 14 is Right Elbow
        l_elbow_raw = np.array([-l_elbow.x, l_elbow.z * 0.5, -l_elbow.y])
        l_elbow_offset = (l_elbow_raw - mid_hip_raw) * SCALE
        bones["upper_arm_ik_target.L"] = l_elbow_offset.tolist()
        
        # Right Elbow Target (Character) <- Left Elbow (User)
        r_elbow = results.pose_landmarks.landmark[13] # 13 is Left Elbow
        r_elbow_raw = np.array([-r_elbow.x, r_elbow.z * 0.5, -r_elbow.y])
        r_elbow_offset = (r_elbow_raw - mid_hip_raw) * SCALE
        bones["upper_arm_ik_target.R"] = r_elbow_offset.tolist()
        
        # Left Knee Target (Character) <- Right Knee (User)
        l_knee = results.pose_landmarks.landmark[26] # 26 is Right Knee
        l_knee_raw = np.array([-l_knee.x, l_knee.z * 0.5, -l_knee.y])
        l_knee_offset = (l_knee_raw - mid_hip_raw) * SCALE
        bones["thigh_ik_target.L"] = l_knee_offset.tolist()
        
        # Right Knee Target (Character) <- Left Knee (User)
        r_knee = results.pose_landmarks.landmark[25] # 25 is Left Knee
        r_knee_raw = np.array([-r_knee.x, r_knee.z * 0.5, -r_knee.y])
        r_knee_offset = (r_knee_raw - mid_hip_raw) * SCALE
        bones["thigh_ik_target.R"] = r_knee_offset.tolist()

        return bones
