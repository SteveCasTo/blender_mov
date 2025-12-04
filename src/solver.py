import numpy as np
from scipy.spatial.transform import Rotation as R
from src.smoothing import MultiChannelSmoother

class PoseSolver:
    def __init__(self):
        # Definir los vectores estándar de Pose en T para un esqueleto humano en el sistema de coordenadas de Blender (Z-arriba, Y-adelante)
        # Asumimos Rigify/Mixamo estándar:
        # Pose en T: Brazos a lo largo del eje X, Piernas a lo largo del eje -Z, Columna a lo largo del eje +Z.
        
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
        self.rest_hand_basis = {}
        self.initial_hip_pos = None
        self.initial_hip_pos = None
        # Suavizado / continuidad para el giro (yaw) para evitar saltos bruscos
        self.torso_yaw_smoothed = 0.0
        self.pelvis_yaw_smoothed = 0.0
        self.prev_torso_raw = None
        self.prev_pelvis_raw = None
        # coeficiente de suavizado (0..1), mayor = más sensible
        self.yaw_smoothing_alpha = 0.25
        
        # Inicializar OneEuroFilter para todos los huesos
        # min_cutoff=1.0: Filtra temblores > 1Hz cuando está quieto
        # beta=0.5: Reduce latencia cuando se mueve rápido
        self.smoother = MultiChannelSmoother(min_cutoff=1.0, beta=0.5)

    def calibrate(self, landmarks):
        """
        Captura la pose actual como la Pose de Descanso (Pose en T).
        Actualiza self.rest_vectors con los vectores de los puntos clave actuales.
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

        # Actualizar Vectores de Descanso basados en el CUERPO REAL DEL USUARIO
        # Columna
        mid_hip = type('P', (), {})()
        mid_hip.x = (converted_lm[23].x + converted_lm[24].x) / 2
        mid_hip.y = (converted_lm[23].y + converted_lm[24].y) / 2
        mid_hip.z = (converted_lm[23].z + converted_lm[24].z) / 2
        
        mid_shoulder = type('P', (), {})()
        mid_shoulder.x = (converted_lm[11].x + converted_lm[12].x) / 2
        mid_shoulder.y = (converted_lm[11].y + converted_lm[12].y) / 2
        mid_shoulder.z = (converted_lm[11].z + converted_lm[12].z) / 2
        
        self.rest_vectors["spine"] = self.get_vector(mid_hip, mid_shoulder)
        
        # Cuello (Mitad de hombros a Nariz)
        if len(converted_lm) > 0:
            nose = converted_lm[0]
            self.rest_vectors["neck"] = self.get_vector(mid_shoulder, nose)

        # Brazos
        self.rest_vectors["left_arm"] = get_calib_vec(11, 13)
        self.rest_vectors["left_forearm"] = get_calib_vec(13, 15)
        self.rest_vectors["right_arm"] = get_calib_vec(12, 14)
        self.rest_vectors["right_forearm"] = get_calib_vec(14, 16)
        
        # Calibración de Base de Manos (si las manos son visibles)
        # Necesitamos puntos clave para las manos. El argumento 'landmarks' pasado a calibrate 
        # suele ser solo pose_landmarks. Podríamos no tener puntos de manos aquí 
        # si el usuario solo está en Pose en T sin seguimiento de manos activo/calibrado.
        # Sin embargo, podemos aproximar usando Muñeca(15/16), Índice(19/20), Meñique(17/18) de los puntos de POSE.
        # MediaPipe Pose tiene:
        # 15: Muñeca Izq, 17: Meñique Izq, 19: Índice Izq
        # 16: Muñeca Der, 18: Meñique Der, 20: Índice Der
        
        # Mano Izquierda (15, 19, 17) -> Muñeca, Índice, Meñique
        self.rest_hand_basis["left"] = self.calculate_basis_rotation(
            converted_lm[15], converted_lm[19], converted_lm[17]
        )
        
        # Mano Derecha (16, 20, 18) -> Muñeca, Índice, Meñique
        self.rest_hand_basis["right"] = self.calculate_basis_rotation(
            converted_lm[16], converted_lm[20], converted_lm[18]
        )
        
        # Piernas
        self.rest_vectors["left_up_leg"] = get_calib_vec(23, 25)
        self.rest_vectors["left_leg"] = get_calib_vec(25, 27)
        self.rest_vectors["right_up_leg"] = get_calib_vec(24, 26)
        self.rest_vectors["right_leg"] = get_calib_vec(26, 28)
        
        # Reiniciar Posición de Cadera
        bx = mid_hip.x
        by = mid_hip.y
        bz = mid_hip.z
        self.initial_hip_pos = np.array([bx, by, bz])
        
        print("Calibration Complete.")

    def calculate_root_translation(self, landmarks):
        """
        Calcula la traslación de la raíz (caderas) relativa a la posición inicial.
        Devuelve un vector [x, y, z] en unidades de Blender.
        """
        # Centro de la Cadera en MediaPipe
        # 23: Cadera Izq, 24: Cadera Der
        # Coordenadas MP: X (0-1), Y (0-1), Z (escala aprox)
        
        # Necesitamos convertir a coordenadas de Blender primero:
        # Blender X = MP X
        # Blender Y = MP Z
        # Blender Z = -MP Y
        
        # Obtener coordenadas MP crudas para caderas
        l_hip = landmarks[23]
        r_hip = landmarks[24]
        
        # Promediar para obtener el centro
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
            
        # Calcular Delta
        delta = current_pos - self.initial_hip_pos
        
        # Factor de Escala
        # Las coordenadas de MediaPipe son pequeñas. Necesitamos escalarlas para Blender.
        # Un humano mide ~1.7m. En MP, la altura es ~0.5-0.8 unidades?
        # Probemos un factor de escala de 2.0 o 3.0 inicialmente.
        SCALE = 3.0
        
        # ¿Invertir X porque MP está espejado?
        # Usualmente la webcam está espejada. Si el usuario se mueve a la izquierda, la imagen se mueve a la derecha.
        # Blender X+ es Derecha.
        # Dejemos X como está por ahora, podría necesitar -X.
        
        translation = delta * SCALE
        
        # Podríamos querer poner a cero Y (Adelante/Atrás) si la profundidad es ruidosa
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
            return np.array([0, 1, 0, 0])

        axis = np.cross(u, v)
        
        dot = np.dot(u, v)
        
        q_xyz = axis
        q_w = np.sqrt(np.linalg.norm(u)**2 * np.linalg.norm(v)**2) + dot
        
        q = np.array([q_w, q_xyz[0], q_xyz[1], q_xyz[2]])
        return self.normalize(q)

    def calculate_basis_rotation(self, origin, p_primary, p_secondary):
        """
        Calcular rotación desde 3 puntos que definen un plano.
        origin: Muñeca
        p_primary: Índice o Medio MCP (define vector Adelante)
        p_secondary: Meñique MCP (define Plano)
        
        Devuelve: Cuaternión [w, x, y, z] representando la orientación de esta base
        """
        v_forward = self.normalize(self.get_vector(origin, p_primary))
        v_temp = self.normalize(self.get_vector(origin, p_secondary))
        
        # Calcular Normal (Arriba/Abajo)
        v_normal = self.normalize(np.cross(v_forward, v_temp))
        
        # Calcular vector Derecha
        # Asegurar sistema de coordenadas de mano derecha: Cross(Y, Z) = X
        # v_forward es eje Y, v_normal es eje Z (aprox)
        v_right = self.normalize(np.cross(v_forward, v_normal))
        
        # Recalcular normal para asegurar ortogonalidad
        v_normal = np.cross(v_right, v_forward)
        
        # Construir Matriz de Rotación (Columnas: X, Y, Z)
        matrix = np.column_stack((v_right, v_forward, v_normal))
        
        # Verificar determinante para evitar error "Non-positive determinant"
        if np.linalg.det(matrix) < 0:
            # Voltear un eje para hacerlo de mano derecha (ej. Normal)
            v_normal = -v_normal
            matrix = np.column_stack((v_right, v_forward, v_normal))
        
        # Convertir a Cuaternión
        try:
            r = R.from_matrix(matrix)
            q = r.as_quat()
            # Scipy devuelve [x, y, z, w], necesitamos [w, x, y, z]
            return np.array([q[3], q[0], q[1], q[2]])
        except ValueError:
            return np.array([1, 0, 0, 0])

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
        Entrada: Objeto de resultados de MediaPipe Holistic (contiene pose_landmarks, left_hand_landmarks, right_hand_landmarks)
        Salida: Diccionario de nombres de huesos a [w, x, y, z] (Rotaciones) o [x, y, z] (Posiciones)
        """
        
        if not results.pose_landmarks:
            return {}

        # Helper class for coordinate conversion
        class Point:
            def __init__(self, x, y, z):
                # MediaPipe: X(0-1 izq-der), Y(0-1 arriba-abajo), Z(profundidad)
                # Blender: X(izq-der), Y(atrás-adelante), Z(abajo-arriba)
                self.x = x      # Sin inversión
                # AMORTIGUAR PROFUNDIDAD (Z en MP -> Y en Blender)
                # Multiplicamos por 0.5 para reducir el efecto de "inclinación" causado por la estimación de profundidad exagerada.
                self.y = z * 0.5 
                self.z = -y     # Altura se convierte en Z (arriba/abajo)

        # 1. Convertir puntos clave de POSE
        pose_lm = []
        for lm in results.pose_landmarks.landmark:
            pose_lm.append(Point(lm.x, lm.y, lm.z))

        # 2. Convertir puntos clave de MANOS (si están disponibles)
        left_hand_lm = []
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                left_hand_lm.append(Point(lm.x, lm.y, lm.z))
                
        right_hand_lm = []
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                right_hand_lm.append(Point(lm.x, lm.y, lm.z))

        bones = {}
        
        # Helper para obtener rotación GLOBAL para un hueso específico (Genérico)
        def calc_global_rot(landmarks_list, p_start_idx, p_end_idx, rest_vec_name):
            if not landmarks_list or p_start_idx >= len(landmarks_list) or p_end_idx >= len(landmarks_list):
                return np.array([1, 0, 0, 0])
                
            start = landmarks_list[p_start_idx]
            end = landmarks_list[p_end_idx]
            current_vec = self.get_vector(start, end)
            
            if rest_vec_name in self.rest_vectors:
                rest_vec = self.rest_vectors[rest_vec_name]
            else:
                # Valores predeterminados de respaldo
                if "left" in rest_vec_name: rest_vec = np.array([1, 0, 0])
                elif "right" in rest_vec_name: rest_vec = np.array([-1, 0, 0])
                else: rest_vec = np.array([0, 0, 1])
                
            return self.rotation_between_vectors(rest_vec, current_vec)

        # --- 1. COLUMNA Y CABEZA (FK) ---
        # Usa puntos clave de POSE
        
        # Calcular puntos medios para estructura corporal
        # CRÍTICO: Estos deben ser estables incluso cuando la cabeza se mueve
        mid_hip = type('P', (), {})()
        mid_hip.x = (pose_lm[23].x + pose_lm[24].x) / 2
        mid_hip.y = (pose_lm[23].y + pose_lm[24].y) / 2
        mid_hip.z = (pose_lm[23].z + pose_lm[24].z) / 2
        
        mid_shoulder = type('P', (), {})()
        mid_shoulder.x = (pose_lm[11].x + pose_lm[12].x) / 2
        mid_shoulder.y = (pose_lm[11].y + pose_lm[12].y) / 2
        mid_shoulder.z = (pose_lm[11].z + pose_lm[12].z) / 2
        
        # --- TORSO (Rotación Principal del Cuerpo) ---
        # Vector de Caderas a Hombros (inclinación del torso)
        spine_vec = self.get_vector(mid_hip, mid_shoulder)
        
        # Calcular inclinación del torso (adelante/atrás) desde vector cadera->hombro
        # IMPORTANTE: Esto NO debe verse afectado por el movimiento de la cabeza
        q_torso_pitch = self.rotation_between_vectors(np.array([0, 0, 1]), spine_vec)
        
        # --- GIRO DEL TORSO (rotación alrededor del eje vertical al girar) ---
        # Usar orientación de línea de hombros para detectar rotación del cuerpo
        # Hombro Izquierdo (11) y Hombro Derecho (12)
        shoulder_vec = self.get_vector(pose_lm[11], pose_lm[12])
        
        # VERIFICACIÓN DE ESTABILIDAD: Validar que el vector de hombros sea razonable
        shoulder_length = np.linalg.norm(shoulder_vec)
        
        # DEBUG: Print shoulder info to detect asymmetry
        print(f"Shoulder vec: [{shoulder_vec[0]:.3f}, {shoulder_vec[1]:.3f}, {shoulder_vec[2]:.3f}], length: {shoulder_length:.3f}")
        
        # Solo calcular giro si la detección de hombros es estable
        if shoulder_length > 0.05 and shoulder_length < 1.5:  # Límites relajados
            # MODO ESPEJO: Invertir componente adelante para que derecha usuario = izquierda personaje
            shoulder_forward = -shoulder_vec[1]  # Componente Y INVERTIDO para espejo
            shoulder_lateral = shoulder_vec[0]   # X component (left/right)

            # Ángulo de giro crudo desde orientación actual de hombros
            yaw_angle_raw = np.arctan2(shoulder_forward, shoulder_lateral)
            # Agregar offset fijo de 180° si tu rig lo requiere
            yaw_raw = yaw_angle_raw + (np.pi / 2) * 2  # +180° offset para neutral en T-pose

            # Continuidad de ángulo (desenvolver) para evitar saltos alrededor del límite ±π
            if self.prev_torso_raw is not None:
                # bring yaw_raw close to prev by adding/subtracting 2π if needed
                diff = yaw_raw - self.prev_torso_raw
                if diff > np.pi:
                    yaw_raw -= 2 * np.pi
                elif diff < -np.pi:
                    yaw_raw += 2 * np.pi
            # inicializar prev si es necesario
            self.prev_torso_raw = yaw_raw if self.prev_torso_raw is None else self.prev_torso_raw

            # Suavizado exponencial (IIR) en ángulo de giro para evitar saltos bruscos
            self.torso_yaw_smoothed = (
                self.yaw_smoothing_alpha * yaw_raw
                + (1.0 - self.yaw_smoothing_alpha) * self.torso_yaw_smoothed
            )

            yaw_angle = self.torso_yaw_smoothed
            # mantener prev raw alineado al valor suavizado para el siguiente desempaquetado
            self.prev_torso_raw = yaw_raw
        else:
            # Hombros inestables - mantener último valor suavizado (congelar)
            yaw_angle = self.torso_yaw_smoothed
        
        # Crear cuaternión de giro (rotación alrededor del eje Z)
        # Cuaternión para rotación en eje Z: [cos(θ/2), 0, 0, sin(θ/2)]
        q_torso_yaw = np.array([
            np.cos(yaw_angle / 2),
            0,
            0,
            np.sin(yaw_angle / 2)
        ])
        
        # Combinar inclinación y giro para rotación total del torso
        # Aplicar giro primero, luego inclinación (el orden importa)
        q_torso_global = self.multiply_quaternions(q_torso_yaw, q_torso_pitch)
        
        # Aplicar a hueso "torso" (Raíz de la cadena de la columna)
        bones["torso_rot"] = q_torso_global.tolist()
        
        # --- CURVATURA DE COLUMNA ---
        # Aplicar una fracción de la rotación del torso a los huesos de la columna para crear una curva.
        # Distribuir rotación de giro a la columna para torsión natural
        q_spine_yaw = self.scale_quaternion_rotation(q_torso_yaw, 0.3)  # 30% of yaw
        q_spine_pitch = self.scale_quaternion_rotation(q_torso_pitch, 0.2)  # 20% of pitch
        q_spine_curve = self.multiply_quaternions(q_spine_yaw, q_spine_pitch)
        
        bones["spine_fk"] = q_spine_curve.tolist()
        bones["spine_fk.001"] = q_spine_curve.tolist()
        
        # --- PELVIS/CADERAS (para anclaje de piernas) ---
        # La pelvis debe tener su propia rotación basada en orientación de CADERA, no torso
        # Esto es crítico: las piernas están unidas a la pelvis, no a la columna
        
        # Calcular giro de pelvis desde orientación de línea de cadera
        hip_vec = self.get_vector(pose_lm[23], pose_lm[24])  # Cadera Izq -> Cadera Der
        
        # VERIFICACIÓN DE ESTABILIDAD: Validar que el vector de cadera sea razonable
        hip_length = np.linalg.norm(hip_vec)
        
        # DEBUG: Print hip info
        print(f"Hip vec: [{hip_vec[0]:.3f}, {hip_vec[1]:.3f}, {hip_vec[2]:.3f}], length: {hip_length:.3f}")
        
        # Solo calcular giro de pelvis si la detección de cadera es estable
        if hip_length > 0.05 and hip_length < 1.5:  # Límites relajados (igual que hombros)
            # MODO ESPEJO: Invertir componente adelante
            hip_forward = -hip_vec[1]  # Componente Y INVERTIDO para espejo
            hip_lateral = hip_vec[0]  # Componente X (izq/der)

            # Giro crudo de pelvis
            pelvis_yaw_raw = np.arctan2(hip_forward, hip_lateral)
            pelvis_yaw_raw = pelvis_yaw_raw + (np.pi / 2) * 2  # Mismo offset de 180° que el torso

            # Desenvolver relativo al previo pelvis raw para mantener continuidad
            if self.prev_pelvis_raw is not None:
                diff_p = pelvis_yaw_raw - self.prev_pelvis_raw
                if diff_p > np.pi:
                    pelvis_yaw_raw -= 2 * np.pi
                elif diff_p < -np.pi:
                    pelvis_yaw_raw += 2 * np.pi

            # Suavizado exponencial para giro de pelvis
            self.pelvis_yaw_smoothed = (
                self.yaw_smoothing_alpha * pelvis_yaw_raw
                + (1.0 - self.yaw_smoothing_alpha) * self.pelvis_yaw_smoothed
            )
            pelvis_yaw = self.pelvis_yaw_smoothed
            self.prev_pelvis_raw = pelvis_yaw_raw
        else:
            # Caderas inestables - mantener último valor suavizado (congelar)
            pelvis_yaw = self.pelvis_yaw_smoothed
        
        # Crear cuaternión de giro de pelvis
        q_pelvis_yaw = np.array([
            np.cos(pelvis_yaw / 2),
            0,
            0,
            np.sin(pelvis_yaw / 2)
        ])
        
        # La pelvis debe tener inclinación mínima (las caderas no se inclinan mucho)
        # Usar cuaternión identidad o fracción muy pequeña de inclinación del torso
        q_pelvis_global = q_pelvis_yaw  # Pelvis solo rota (giro), inclinación mínima

        # --- CABEZA Y CUELLO ---
        # Calcular Rotación de Cabeza y Cuello con manejo adecuado de inclinación (arriba/abajo)
        # MODO ESPEJO: Movimientos del usuario espejados (Izq Usuario = Der Personaje)
        
        if len(pose_lm) > 0:
            # Usar NARIZ (se mueve más que las orejas al girar la cabeza)
            nose = pose_lm[0]
            l_ear = pose_lm[7]
            r_ear = pose_lm[8]
            l_eye = pose_lm[2]
            r_eye = pose_lm[5]
            
            # Mitad de orejas para referencia
            mid_ear = type('P', (), {})()
            mid_ear.x = (l_ear.x + r_ear.x) / 2
            mid_ear.y = (l_ear.y + r_ear.y) / 2
            mid_ear.z = (l_ear.z + r_ear.z) / 2
            
            # Mitad de ojos para mejor detección de inclinación
            mid_eye = type('P', (), {})()
            mid_eye.x = (l_eye.x + r_eye.x) / 2
            mid_eye.y = (l_eye.y + r_eye.y) / 2
            mid_eye.z = (l_eye.z + r_eye.z) / 2
            
            # --- DETECTAR INCLINACIÓN (tilt arriba/abajo) ---
            # Al mirar abajo: nose.z < eye.z (inclinación negativa)
            # Al mirar arriba: nose.z > eye.z (inclinación positiva)
            pitch_raw = (nose.z - mid_eye.z) * 8.0  # Sensibilidad reducida
            
            # LIMITAR inclinación a rango realista: -30° a +20°
            pitch_amount = np.clip(pitch_raw, -30.0, 20.0)
            
            # --- ROTACIÓN DE CUELLO (absorbe parte de la inclinación) ---
            # Vector base de cuello: hombro a mitad de orejas
            neck_base_vec = self.get_vector(mid_shoulder, mid_ear)
            
            # Aplicar inclinación al cuello (cuello se inclina adelante/atrás al mirar abajo/arriba)
            # Aumentar contribución del cuello ligeramente para que sea visible al mirar abajo
            neck_pitch_factor = pitch_amount * 0.35

            # Ajustar vector de cuello con inclinación
            # Cuando la inclinación es negativa (mirar abajo), inclinar cuello adelante (aumentar Y)
            # Usar un multiplicador ligeramente mayor para hacer que el cuello se flexione visiblemente
            neck_vec = np.array([
                neck_base_vec[0],
                neck_base_vec[1] + neck_pitch_factor * 0.05,
                neck_base_vec[2]
            ])
            neck_vec = self.normalize(neck_vec)
            
            # Calcular rotación de cuello
            q_neck_global = self.rotation_between_vectors(np.array([0, 0, 1]), neck_vec)
            
            # Cuello es hijo del torso
            q_neck_local = self.multiply_quaternions(self.invert_quaternion(q_torso_global), q_neck_global)
            bones["neck"] = q_neck_local.tolist()
            
            # --- ROTACIÓN DE CABEZA (hija del cuello, absorbe inclinación restante) ---
            # Calcular GIRO (izq/der): Usar posición horizontal de nariz relativa al centro de la cara
            # MODO ESPEJO: Invertir X para efecto espejo
            yaw_raw = -(nose.x - mid_eye.x) * 50.0  # Reducido de 70.0
            yaw_diff = np.clip(yaw_raw, -80.0, 80.0)  # Limit left/right rotation
            
            # Construir un vector objetivo físico de cuello a nariz (esto representa dónde debería mirar la cabeza)
            head_target_vec = self.get_vector(mid_ear, nose)
            # Espejar X para modo espejo (usamos giro espejado previamente)
            head_target_vec[0] = -head_target_vec[0]
            
            # ARREGLO CRÍTICO (revisado): rotar el objetivo para que neutral = mirar al frente.
            # Mapeo anterior rotaba en dirección opuesta; usar rotación hacia adelante en su lugar.
            # Mapeo: Original: [X, Y_adelante, Z_arriba] -> Corregido: [X, Z, -Y]
            # Esto rota el componente adelante hacia el eje arriba en la dirección adelante.
            head_target_corrected = np.array([
                head_target_vec[0],      # Keep X (left/right) as-is
                head_target_vec[2],      # Y becomes Z (use up component as forward)
                -head_target_vec[1]      # Z becomes -Y (forward becomes up with sign inverted)
            ])
            head_target_corrected = self.normalize(head_target_corrected)

            # Calcular rotación global de cabeza alineando cabeza +Z al vector objetivo
            q_head_global = self.rotation_between_vectors(np.array([0, 0, 1]), head_target_corrected)
            
            # Cabeza es hija de CUELLO
            # Calcular rotación local relativa al cuello
            q_head_local = self.multiply_quaternions(self.invert_quaternion(q_neck_global), q_head_global)

            bones["head_fk"] = q_head_local.tolist()
        else:
            bones["head_fk"] = [1, 0, 0, 0]
            bones["neck"] = [1, 0, 0, 0]
            q_head_global = q_torso_global # Fallback

        # --- 2. BRAZOS (FK) ---
        # MODO ESPEJO: Izquierda Usuario -> Derecha Personaje, Derecha Usuario -> Izquierda Personaje
        
        # Necesitamos usar q_torso_global como la rotación padre para los brazos
        # porque los brazos están unidos al torso (vía hombro/clavícula).
        q_spine_global = q_torso_global # Alias para compatibilidad con lógica de brazos existente
        
        # Brazo Izquierdo (Personaje) <- Brazo Derecho (Usuario)
        # Puntos Clave: 12 (Hombro Der), 14 (Codo Der), 16 (Muñeca Der)
        q_upper_arm_L_global = calc_global_rot(pose_lm, 12, 14, "right_arm")
        q_forearm_L_global = calc_global_rot(pose_lm, 14, 16, "right_forearm")
        
        q_upper_arm_L_local = self.multiply_quaternions(self.invert_quaternion(q_spine_global), q_upper_arm_L_global)
        bones["upper_arm_fk.L"] = q_upper_arm_L_local.tolist()
        
        q_upper_arm_L_inv = self.invert_quaternion(q_upper_arm_L_global)
        q_forearm_L_local = self.multiply_quaternions(q_upper_arm_L_inv, q_forearm_L_global)
        bones["forearm_fk.L"] = q_forearm_L_local.tolist()
        
        # Brazo Derecho (Personaje) <- Brazo Izquierdo (Usuario)
        # Puntos Clave: 11 (Hombro Izq), 13 (Codo Izq), 15 (Muñeca Izq)
        q_upper_arm_R_global = calc_global_rot(pose_lm, 11, 13, "left_arm")
        q_forearm_R_global = calc_global_rot(pose_lm, 13, 15, "left_forearm")
        
        q_upper_arm_R_local = self.multiply_quaternions(self.invert_quaternion(q_spine_global), q_upper_arm_R_global)
        bones["upper_arm_fk.R"] = q_upper_arm_R_local.tolist()
        
        q_upper_arm_R_inv = self.invert_quaternion(q_upper_arm_R_global)
        q_forearm_R_local = self.multiply_quaternions(q_upper_arm_R_inv, q_forearm_R_global)
        bones["forearm_fk.R"] = q_forearm_R_local.tolist()

        # --- 3. PIERNAS (FK) ---
        # MODO ESPEJO
        
        # Pierna Izquierda (Personaje) <- Pierna Derecha (Usuario)
        # Puntos Clave: 24 (Cadera Der), 26 (Rodilla Der), 28 (Tobillo Der)
        q_thigh_L_global = calc_global_rot(pose_lm, 24, 26, "right_up_leg")
        q_shin_L_global = calc_global_rot(pose_lm, 26, 28, "right_leg")
        
        # Usar pelvis (solo inclinación) en lugar de columna (inclinación+giro) para prevenir problemas de rotación
        q_thigh_L_local = self.multiply_quaternions(self.invert_quaternion(q_pelvis_global), q_thigh_L_global)
        bones["thigh_fk.L"] = q_thigh_L_local.tolist()
        
        q_thigh_L_inv = self.invert_quaternion(q_thigh_L_global)
        q_shin_L_local = self.multiply_quaternions(q_thigh_L_inv, q_shin_L_global)
        bones["shin_fk.L"] = q_shin_L_local.tolist()
        
        # Pierna Derecha (Personaje) <- Pierna Izquierda (Usuario)
        # Puntos Clave: 23 (Cadera Izq), 25 (Rodilla Izq), 27 (Tobillo Izq)
        q_thigh_R_global = calc_global_rot(pose_lm, 23, 25, "left_up_leg")
        q_shin_R_global = calc_global_rot(pose_lm, 25, 27, "left_leg")
        
        # Usar pelvis (solo inclinación) en lugar de columna (inclinación+giro) para prevenir problemas de rotación
        q_thigh_R_local = self.multiply_quaternions(self.invert_quaternion(q_pelvis_global), q_thigh_R_global)
        bones["thigh_fk.R"] = q_thigh_R_local.tolist()
        
        q_thigh_R_inv = self.invert_quaternion(q_thigh_R_global)
        q_shin_R_local = self.multiply_quaternions(q_thigh_R_inv, q_shin_R_global)
        bones["shin_fk.R"] = q_shin_R_local.tolist()

        # --- 4. DEDOS (FK) ---
        # MODO ESPEJO
        
        def solve_finger_chain(lm_list, prefix, indices, parent_global_rot, side_name, palm_normal):
            if not lm_list: return
            
            # Helper para crear un Punto desde coordenadas
            class TempPoint:
                def __init__(self, x, y, z):
                    self.x, self.y, self.z = x, y, z

            # 1. Proximal (MCP -> PIP)
            # Construir un 3er punto para definir el plano (Inicio + Normal de Palma)
            # Esto fuerza al vector "Arriba" del dedo a alinearse con la Normal de la Palma
            start = lm_list[indices[0]]
            p_normal_target = TempPoint(start.x + palm_normal[0], start.y + palm_normal[1], start.z + palm_normal[2])
            
            q_prox_global = self.calculate_basis_rotation(start, lm_list[indices[1]], p_normal_target)
            q_prox_local = self.multiply_quaternions(self.invert_quaternion(parent_global_rot), q_prox_global)
            bones[f"{prefix}.01.{side_name[0].upper()}"] = q_prox_local.tolist()
            
            # 2. Intermedia (PIP -> DIP)
            start = lm_list[indices[1]]
            p_normal_target = TempPoint(start.x + palm_normal[0], start.y + palm_normal[1], start.z + palm_normal[2])
            
            q_inter_global = self.calculate_basis_rotation(start, lm_list[indices[2]], p_normal_target)
            q_inter_local = self.multiply_quaternions(self.invert_quaternion(q_prox_global), q_inter_global)
            bones[f"{prefix}.02.{side_name[0].upper()}"] = q_inter_local.tolist()
            
            # 3. Distal (DIP -> Punta)
            start = lm_list[indices[2]]
            p_normal_target = TempPoint(start.x + palm_normal[0], start.y + palm_normal[1], start.z + palm_normal[2])
            
            q_dist_global = self.calculate_basis_rotation(start, lm_list[indices[3]], p_normal_target)
            q_dist_local = self.multiply_quaternions(self.invert_quaternion(q_inter_global), q_dist_global)
            bones[f"{prefix}.03.{side_name[0].upper()}"] = q_dist_local.tolist()

        # Mano Izquierda (Personaje) <- Mano Derecha (Usuario)
        if right_hand_lm:
            # Calcular Rotación Global de Mano usando 3 puntos (Muñeca, Índice, Meñique)
            # 0: Muñeca, 5: Índice MCP, 17: Meñique MCP
            q_hand_L_global = self.calculate_basis_rotation(right_hand_lm[0], right_hand_lm[5], right_hand_lm[17])
            
            # Calculate relative rotation from Rest Pose
            # Character Left Hand corresponds to User Right Hand Rest Basis
            q_rest = self.rest_hand_basis.get("right", np.array([1, 0, 0, 0]))
            q_diff = self.multiply_quaternions(q_hand_L_global, self.invert_quaternion(q_rest))
            
            # Apply to parent (Forearm)
            # But wait, q_diff is the GLOBAL rotation difference.
            # We need to apply this relative to the Forearm's current global rotation?
            # Or just replace the hand rotation?
            # The bone rotation in Blender is Local.
            # Local = Parent_Global_Inv * Child_Global
            
            # We constructed q_hand_L_global as an absolute orientation in Blender space.
            # However, our basis construction might not align perfectly with the bone axes if T-Pose wasn't perfect.
            # That's why we use q_diff (Rotation from Rest).
            # Global_Current = Rest_Global * q_diff (roughly)
            # Actually, let's try using q_hand_L_global directly if we trust the basis mapping (Y=Forward).
            # If we use q_diff, we are saying "Rotate the hand by X degrees from T-pose".
            
            # Let's try the standard Local conversion:
            # q_hand_L_local = inv(q_forearm_L_global) * q_hand_L_global
            
            # But we need q_hand_L_global to be correct relative to the Rest Pose.
            # If we use calculate_basis_rotation, we get the absolute orientation of the triangle 0-5-17.
            # In T-Pose, this triangle has a specific orientation (Rest Basis).
            # We want the Bone to have the same rotation relative to its Rest Pose.
            
            # Let's use the delta rotation approach which is more robust to rig differences:
            # 1. Calculate Delta: Q_delta = Q_current * inv(Q_rest)
            # 2. Apply Delta to the known Rest Vector of the arm?
            # No, simpler:
            # The Bone's Global Rotation should be: Q_bone_global = Q_delta * Q_bone_rest_global
            # Q_bone_rest_global is roughly "Left Arm Out".
            
            # Let's try the direct basis mapping first, it's cleaner if it works.
            # We assume the bone Y axis points from Wrist to Middle Finger.
            # Our basis Y axis points from Wrist to Index (or Middle).
            # Let's use Middle (9) for Forward to match bone axis better.
            q_hand_L_global = self.calculate_basis_rotation(right_hand_lm[0], right_hand_lm[9], right_hand_lm[17])
            
            # Fix for Mirroring? 
            # User Right Hand (Basis) -> Character Left Hand.
            # If we just map the basis, X might be inverted.
            # Basis: Y=Forward, Z=Up, X=Right.
            # User Right Hand: Y=Left, Z=Back, X=Up?
            # Let's rely on the calibration delta, it handles the coordinate system transform implicitly.
            
            q_current = q_hand_L_global
            q_rest = self.rest_hand_basis.get("right", np.array([1, 0, 0, 0]))
            
            # Delta: Rotation from Rest to Current
            q_delta = self.multiply_quaternions(q_current, self.invert_quaternion(q_rest))
            
            # Apply Delta to the standard "Left Arm" orientation (Identity/Rest)
            # Character Left Hand Rest: Points +X.
            # We want to rotate it by q_delta.
            # New Global = q_delta * Rest_Orientation
            # Rest Orientation for Left Hand is usually just Identity (if parent is aligned) or +X.
            # Actually, let's just use the Local calculation:
            # Local = Inv(Parent_Global) * Current_Global
            # We need Current_Global to be correct for the Character.
            
            # If we assume the User's Hand Basis IS the Character's Hand Basis (mirrored),
            # Then q_hand_L_global (calculated from User Right Hand) needs to be mirrored?
            # Mirroring a quaternion basis is tricky.
            
            # Alternative: Use the "Rotation Between Vectors" for the main axis (as before)
            # AND add the Twist rotation.
            # But 3-point basis is better.
            
            # Let's try applying q_delta to the Character's Rest Pose.
            # Character Left Hand Rest Global: Points +X (Left).
            # We need a quaternion that represents "Pointing Left".
            # If we assume the rig is standard, the Global Rotation of the hand in T-Pose is...
            # It depends on the bone roll.
            
            # Let's stick to the Local conversion which is mathematically sound:
            # bones["hand_fk.L"] = Inv(Forearm_Global) * Hand_Global
            # We just need Hand_Global to be correct.
            # Hand_Global should be: The orientation of the user's hand, mapped to character.
            # User Right Hand -> Character Left Hand.
            # We need to mirror the basis.
            # User Right: X, Y, Z.
            # Character Left: -X, Y, Z? (Mirror X)
            
            # Let's try a simple mapping first:
            # Use the q_delta approach.
            # q_hand_L_global = q_delta * q_forearm_L_global (approx)
            # No, that assumes hand follows forearm exactly plus delta.
            
            # Let's go with:
            # 1. Calculate User Right Hand Rotation (Absolute)
            # 2. Calculate User Right Forearm Rotation (Absolute)
            # 3. Calculate Local Rotation (Wrist relative to Forearm)
            # 4. Apply this Local Rotation to Character Left Hand.
            
            # Antebrazo Derecho Usuario Global
            q_u_forearm_global = calc_global_rot(pose_lm, 14, 16, "right_forearm")
            
            # Mano Derecha Usuario Global
            q_u_hand_global = self.calculate_basis_rotation(right_hand_lm[0], right_hand_lm[9], right_hand_lm[17])
            
            # Muñeca Usuario Local (Mano relativa a Antebrazo)
            q_wrist_local = self.multiply_quaternions(self.invert_quaternion(q_u_forearm_global), q_u_hand_global)
            
            # Aplicar a Mano Izquierda Personaje
            # Dado que es espejado, podríamos necesitar invertir algunos ejes.
            # Pero usualmente, "Doblar Arriba" es "Doblar Arriba" en ambos lados.
            # "Girar Adentro" es "Girar Adentro".
            # ¡Así que la Rotación Local podría ser directamente aplicable!
            
            bones["hand_fk.L"] = q_wrist_local.tolist()
            
            # Calcular Normal de Palma (Eje Z de la Base de la Mano)
            # q_hand_L_global es [w, x, y, z]
            # Scipy espera [x, y, z, w]
            r_hand = R.from_quat([q_hand_L_global[1], q_hand_L_global[2], q_hand_L_global[3], q_hand_L_global[0]])
            # En nuestra base, Z es Normal.
            # Invertir para hacer que los dedos se curven hacia la palma (no hacia afuera)
            palm_normal = -r_hand.apply([0, 0, 1])
            
            # Dedos
            solve_finger_chain(right_hand_lm, "thumb", [1, 2, 3, 4], q_hand_L_global, "left", palm_normal)
            solve_finger_chain(right_hand_lm, "f_index", [5, 6, 7, 8], q_hand_L_global, "left", palm_normal)
            solve_finger_chain(right_hand_lm, "f_middle", [9, 10, 11, 12], q_hand_L_global, "left", palm_normal)
            solve_finger_chain(right_hand_lm, "f_ring", [13, 14, 15, 16], q_hand_L_global, "left", palm_normal)
            solve_finger_chain(right_hand_lm, "f_pinky", [17, 18, 19, 20], q_hand_L_global, "left", palm_normal)
            
        # Mano Derecha (Personaje) <- Mano Izquierda (Usuario)
        if left_hand_lm:
            # Antebrazo Izquierdo Usuario Global
            q_u_forearm_global = calc_global_rot(pose_lm, 13, 15, "left_forearm")
            
            # Mano Izquierda Usuario Global
            q_hand_R_global = self.calculate_basis_rotation(left_hand_lm[0], left_hand_lm[9], left_hand_lm[17])
            
            # Muñeca Usuario Local
            q_wrist_local = self.multiply_quaternions(self.invert_quaternion(q_u_forearm_global), q_hand_R_global)
            
            bones["hand_fk.R"] = q_wrist_local.tolist()
            
            # Calcular Normal de Palma
            r_hand = R.from_quat([q_hand_R_global[1], q_hand_R_global[2], q_hand_R_global[3], q_hand_R_global[0]])
            # Invertir para hacer que los dedos se curven hacia la palma (no hacia afuera)
            palm_normal = -r_hand.apply([0, 0, 1])
            
            # Dedos
            solve_finger_chain(left_hand_lm, "thumb", [1, 2, 3, 4], q_hand_R_global, "right", palm_normal)
            solve_finger_chain(left_hand_lm, "f_index", [5, 6, 7, 8], q_hand_R_global, "right", palm_normal)
            solve_finger_chain(left_hand_lm, "f_middle", [9, 10, 11, 12], q_hand_R_global, "right", palm_normal)
            solve_finger_chain(left_hand_lm, "f_ring", [13, 14, 15, 16], q_hand_R_global, "right", palm_normal)
            solve_finger_chain(left_hand_lm, "f_pinky", [17, 18, 19, 20], q_hand_R_global, "right", palm_normal)


        # --- 5. OBJETIVOS IK (Modo Híbrido) ---
        # MODO ESPEJO
        
        # Traslación de Raíz / Caderas
        bones["hips"] = self.calculate_root_translation(results.pose_landmarks.landmark)
        
        # Obtener centro de cadera como referencia
        mid_hip_pos = np.array([mid_hip.x, mid_hip.y, mid_hip.z])
        
        # Factor de escala (calibrado para tamaño de personaje ~7 unidades de alto)
        SCALE = 14.0
        
        # mid_hip_raw para consistencia de conversión de coordenadas (Promedio de Cadera Izq y Der)
        l_hip = results.pose_landmarks.landmark[23]
        r_hip = results.pose_landmarks.landmark[24]
        mid_hip_raw = np.array([
            -(l_hip.x + r_hip.x) / 2, 
            ((l_hip.z + r_hip.z) / 2) * 0.5, # AMORTIGUAR PROFUNDIDAD
            -(l_hip.y + r_hip.y) / 2
        ])
        
        # 1. Manos (Posición de Muñeca)
        l_wrist = results.pose_landmarks.landmark[16]
        l_hand_raw = np.array([-l_wrist.x, l_wrist.z * 0.5, -l_wrist.y])
        l_hand_offset = (l_hand_raw - mid_hip_raw) * SCALE
        bones["hand_ik.L"] = l_hand_offset.tolist()
        
        # Mano Derecha (Personaje) <- Muñeca Izquierda (Usuario)
        r_wrist = results.pose_landmarks.landmark[15]
        r_hand_raw = np.array([-r_wrist.x, r_wrist.z * 0.5, -r_wrist.y])
        r_hand_offset = (r_hand_raw - mid_hip_raw) * SCALE
        bones["hand_ik.R"] = r_hand_offset.tolist()
        
        # 2. Pies (Posición de Tobillo)
        # Pie Izquierdo (Personaje) <- Tobillo Derecho (Usuario)
        l_ankle = results.pose_landmarks.landmark[28]
        l_foot_raw = np.array([-l_ankle.x, l_ankle.z * 0.5, -l_ankle.y])
        l_foot_offset = (l_foot_raw - mid_hip_raw) * SCALE
        bones["foot_ik.L"] = l_foot_offset.tolist()
        
        # Pie Derecho (Personaje) <- Tobillo Izquierdo (Usuario)
        r_ankle = results.pose_landmarks.landmark[27]
        r_foot_raw = np.array([-r_ankle.x, r_ankle.z * 0.5, -r_ankle.y])
        r_foot_offset = (r_foot_raw - mid_hip_raw) * SCALE
        bones["foot_ik.R"] = r_foot_offset.tolist()
        
        # 3. Vectores Polares (Codos y Rodillas)
        l_elbow = results.pose_landmarks.landmark[14]
        l_elbow_raw = np.array([-l_elbow.x, l_elbow.z * 0.5, -l_elbow.y])
        l_elbow_offset = (l_elbow_raw - mid_hip_raw) * SCALE
        bones["upper_arm_ik_target.L"] = l_elbow_offset.tolist()
        
        r_elbow = results.pose_landmarks.landmark[13]
        r_elbow_raw = np.array([-r_elbow.x, r_elbow.z * 0.5, -r_elbow.y])
        r_elbow_offset = (r_elbow_raw - mid_hip_raw) * SCALE
        bones["upper_arm_ik_target.R"] = r_elbow_offset.tolist()
        
        l_knee = results.pose_landmarks.landmark[26]
        l_knee_raw = np.array([-l_knee.x, l_knee.z * 0.5, -l_knee.y])
        l_knee_offset = (l_knee_raw - mid_hip_raw) * SCALE
        bones["thigh_ik_target.L"] = l_knee_offset.tolist()
        
        r_knee = results.pose_landmarks.landmark[25]
        r_knee_raw = np.array([-r_knee.x, r_knee.z * 0.5, -r_knee.y])
        r_knee_offset = (r_knee_raw - mid_hip_raw) * SCALE
        bones["thigh_ik_target.R"] = r_knee_offset.tolist()

        # --- 6. SUAVIZADO ---
        # Aplicar OneEuroFilter a todos los datos de huesos
        smoothed_bones = self.smoother.update(bones)
        
        return smoothed_bones
