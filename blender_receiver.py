
import bpy
import socket
import json
import mathutils

# --- CONFIGURATION ---
PORT = 9000
UDP_IP = "127.0.0.1" # Listen on localhost
UDP_PORT = PORT # Use the same port for UDP
# Map JSON keys (from solver.py) to Blender Bone Names
# Update the values on the right to match your specific Rig
BONE_MAPPING = {
    # --- FK SPINE & HEAD ---
    # --- FK SPINE & HEAD ---
    "spine_fk": "spine_fk",
    "spine_fk.001": "spine_fk.001",
    "neck": "neck",
    "head_fk": "head", # Rigify usually names the head control "head"
    
    # --- FK ARMS ---
    "upper_arm_fk.L": "upper_arm_fk.L",
    "forearm_fk.L": "forearm_fk.L",
    "hand_fk.L": "hand_fk.L",
    "upper_arm_fk.R": "upper_arm_fk.R",
    "forearm_fk.R": "forearm_fk.R",
    "hand_fk.R": "hand_fk.R",
    
    # --- FK LEGS ---
    "thigh_fk.L": "thigh_fk.L",
    "shin_fk.L": "shin_fk.L",
    "foot_fk.L": "foot_fk.L",
    "thigh_fk.R": "thigh_fk.R",
    "shin_fk.R": "shin_fk.R",
    "foot_fk.R": "foot_fk.R",
    
    # --- FK FINGERS (LEFT) ---
    "thumb.01.L": "thumb.01.L", "thumb.02.L": "thumb.02.L", "thumb.03.L": "thumb.03.L",
    "f_index.01.L": "f_index.01.L", "f_index.02.L": "f_index.02.L", "f_index.03.L": "f_index.03.L",
    "f_middle.01.L": "f_middle.01.L", "f_middle.02.L": "f_middle.02.L", "f_middle.03.L": "f_middle.03.L",
    "f_ring.01.L": "f_ring.01.L", "f_ring.02.L": "f_ring.02.L", "f_ring.03.L": "f_ring.03.L",
    "f_pinky.01.L": "f_pinky.01.L", "f_pinky.02.L": "f_pinky.02.L", "f_pinky.03.L": "f_pinky.03.L",

    # --- FK FINGERS (RIGHT) ---
    "thumb.01.R": "thumb.01.R", "thumb.02.R": "thumb.02.R", "thumb.03.R": "thumb.03.R",
    "f_index.01.R": "f_index.01.R", "f_index.02.R": "f_index.02.R", "f_index.03.R": "f_index.03.R",
    "f_middle.01.R": "f_middle.01.R", "f_middle.02.R": "f_middle.02.R", "f_middle.03.R": "f_middle.03.R",
    "f_ring.01.R": "f_ring.01.R", "f_ring.02.R": "f_ring.02.R", "f_ring.03.R": "f_ring.03.R",
    "f_pinky.01.R": "f_pinky.01.R", "f_pinky.02.R": "f_pinky.02.R", "f_pinky.03.R": "f_pinky.03.R",

    # --- IK CONTROLS (Still used for Hybrid/IK mode) ---
    "hand_ik.L": "hand_ik.L",
    "hand_ik.R": "hand_ik.R",
    "upper_arm_ik_target.L": "upper_arm_ik_target.L",
    "upper_arm_ik_target.R": "upper_arm_ik_target.R",
    "foot_ik.L": "foot_ik.L",
    "foot_ik.R": "foot_ik.R",
    "thigh_ik_target.L": "thigh_ik_target.L",
    "thigh_ik_target.R": "thigh_ik_target.R",
    "thigh_ik_target.L": "thigh_ik_target.L",
    "thigh_ik_target.R": "thigh_ik_target.R",
    "torso_loc": "torso", # Translation
    "torso_rot": "torso", # Rotation
}
# ---------------------

def setup_rigify(obj):
    """
    Sets up Rigify properties.
    Note: We default to IK (0.0) but user can slide to FK (1.0) to see green bones.
    """
    print("Setting up Rigify...")
    target_bones = [
        "upper_arm_parent.L", "upper_arm_parent.R",
        "thigh_parent.L", "thigh_parent.R",
    ]
    for bone_name in target_bones:
        if bone_name in obj.pose.bones:
            pbone = obj.pose.bones[bone_name]
            # Default to IK, but don't force it every frame
            if "IK_FK" in pbone:
                pbone["IK_FK"] = 0.0 

class MocapReceiverOperator(bpy.types.Operator):
    """Modal Operator to receive UDP data and animate rig"""
    bl_idname = "wm.mocap_receiver"
    bl_label = "Mocap Receiver"
    
    _timer = None
    sock = None
    
    def modal(self, context, event):
        if event.type == 'TIMER':
            try:
                # Non-blocking receive
                while True:
                    data, addr = self.sock.recvfrom(8192) # Increased buffer for full body data
                    text = data.decode('utf-8')
                    json_data = json.loads(text)
                    
                    # Apply pose
                    obj = context.object
                    if obj and obj.type == 'ARMATURE':
                        self.apply_pose(obj, json_data)
                        
            except socket.error:
                pass # No data
            except Exception as e:
                print(f"Error: {e}")
                
        elif event.type == 'ESC':
            return self.cancel(context)
            
        return {'PASS_THROUGH'}
    
    def apply_pose(self, obj, data):
        # We NO LONGER force IK_FK here. 
        # This allows the user to use the Rigify slider to mix IK and FK.
        
        for bone_key, value in data.items():
            target_bone_name = BONE_MAPPING.get(bone_key)
            if target_bone_name and target_bone_name in obj.pose.bones:
                pbone = obj.pose.bones[target_bone_name]
                
                # Check if value is rotation (4 items) or location (3 items)
                if len(value) == 4:
                    # Rotation (Quaternion)
                    quat = mathutils.Quaternion(value)
                    pbone.rotation_mode = 'QUATERNION'
                    pbone.rotation_quaternion = quat
                elif len(value) == 3:
                    # Location (Translation)
                    # For IK controls, we use Matrix assignment to set absolute position
                    if "_ik" in bone_key and "torso" in obj.pose.bones and "target" not in bone_key:
                        # Note: Pole targets (elbow/knee) are also _ik_target, but usually parented to root or foot.
                        # Ideally we treat them same as hands if they are in world space.
                        # Our solver sends ALL IK targets in world space relative to torso start.
                        
                        # 1. Get Torso (Hip) Position in Object Space
                        torso = obj.pose.bones["torso"]
                        torso_pos = torso.matrix.translation
                        
                        # 2. Calculate Target Position (Torso + Offset)
                        offset = mathutils.Vector(value)
                        target_pos = torso_pos + offset
                        
                        # 3. Apply to Bone Matrix
                        new_matrix = pbone.matrix.copy()
                        new_matrix.translation = target_pos
                        pbone.matrix = new_matrix
                    else:
                        # For non-IK bones (or if torso missing), use direct local assignment
                        pbone.location = mathutils.Vector(value)

    def execute(self, context):
        obj = context.object
        if not obj or obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Active object must be an Armature")
            return {'CANCELLED'}
            
        # Setup Rigify
        setup_rigify(obj)
            
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.016, window=context.window) # ~60 FPS
        wm.modal_handler_add(self)
        
        # Setup UDP
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sock.bind((UDP_IP, UDP_PORT))
        
        print(f"Listening on {UDP_IP}:{UDP_PORT}")
        return {'RUNNING_MODAL'}
    
    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        if self.sock:
            self.sock.close()
        print("Mocap Receiver Stopped")
        return {'CANCELLED'}

def register():
    bpy.utils.register_class(MocapReceiverOperator)

def unregister():
    bpy.utils.unregister_class(MocapReceiverOperator)

if __name__ == "__main__":
    register()
    # Automatically start
    bpy.ops.wm.mocap_receiver()
