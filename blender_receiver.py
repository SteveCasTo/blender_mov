
import bpy
import socket
import json
import mathutils

# --- CONFIGURATION ---
PORT = 9000
UDP_IP = "127.0.0.1"
UDP_PORT = PORT
BONE_MAPPING = {
    "spine_fk": "spine_fk",
    "spine_fk.001": "spine_fk.001",
    "neck": "neck",
    "head_fk": "head",
    
    "upper_arm_fk.L": "upper_arm_fk.L",
    "forearm_fk.L": "forearm_fk.L",
    "hand_fk.L": "hand_fk.L",
    "upper_arm_fk.R": "upper_arm_fk.R",
    "forearm_fk.R": "forearm_fk.R",
    "hand_fk.R": "hand_fk.R",
    
    "thigh_fk.L": "thigh_fk.L",
    "shin_fk.L": "shin_fk.L",
    "foot_fk.L": "foot_fk.L",
    "thigh_fk.R": "thigh_fk.R",
    "shin_fk.R": "shin_fk.R",
    "foot_fk.R": "foot_fk.R",
    
    "thumb.01.L": "thumb.01.L", "thumb.02.L": "thumb.02.L", "thumb.03.L": "thumb.03.L",
    "f_index.01.L": "f_index.01.L", "f_index.02.L": "f_index.02.L", "f_index.03.L": "f_index.03.L",
    "f_middle.01.L": "f_middle.01.L", "f_middle.02.L": "f_middle.02.L", "f_middle.03.L": "f_middle.03.L",
    "f_ring.01.L": "f_ring.01.L", "f_ring.02.L": "f_ring.02.L", "f_ring.03.L": "f_ring.03.L",
    "f_pinky.01.L": "f_pinky.01.L", "f_pinky.02.L": "f_pinky.02.L", "f_pinky.03.L": "f_pinky.03.L",

    "thumb.01.R": "thumb.01.R", "thumb.02.R": "thumb.02.R", "thumb.03.R": "thumb.03.R",
    "f_index.01.R": "f_index.01.R", "f_index.02.R": "f_index.02.R", "f_index.03.R": "f_index.03.R",
    "f_middle.01.R": "f_middle.01.R", "f_middle.02.R": "f_middle.02.R", "f_middle.03.R": "f_middle.03.R",
    "f_ring.01.R": "f_ring.01.R", "f_ring.02.R": "f_ring.02.R", "f_ring.03.R": "f_ring.03.R",
    "f_pinky.01.R": "f_pinky.01.R", "f_pinky.02.R": "f_pinky.02.R", "f_pinky.03.R": "f_pinky.03.R",

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
    "torso_loc": "torso",
    "torso_rot": "torso",
}

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
                while True:
                    data, addr = self.sock.recvfrom(8192)
                    text = data.decode('utf-8')
                    json_data = json.loads(text)
                    
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
        for bone_key, value in data.items():
            target_bone_name = BONE_MAPPING.get(bone_key)
            if target_bone_name and target_bone_name in obj.pose.bones:
                pbone = obj.pose.bones[target_bone_name]
                
                if len(value) == 4:
                    quat = mathutils.Quaternion(value)
                    pbone.rotation_mode = 'QUATERNION'
                    pbone.rotation_quaternion = quat
                elif len(value) == 3:
                    if "_ik" in bone_key and "torso" in obj.pose.bones and "target" not in bone_key:
                        
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
    bpy.ops.wm.mocap_receiver()
