import bpy
import json
import os
import mathutils

def analyze_rig():
    obj = bpy.context.object
    if not obj or obj.type != 'ARMATURE':
        print("ERROR: Active object is not an Armature!")
        return

    rig_info = {
        "armature_name": obj.name,
        "bones": {}
    }

    print(f"Analyzing Armature: {obj.name}...")

    for bone in obj.data.bones:
        pbone = obj.pose.bones[bone.name]
        
        rest_vec = (bone.tail_local - bone.head_local).normalized()
        
        matrix_local = [list(row) for row in bone.matrix_local]
        
        bone_data = {
            "name": bone.name,
            "parent": bone.parent.name if bone.parent else None,
            "length": bone.length,
            "head_local": list(bone.head_local),
            "tail_local": list(bone.tail_local),
            "vector": list(rest_vec),
            "matrix_local": matrix_local,
            "rotation_mode": pbone.rotation_mode,
            "is_deform": bone.use_deform,
            "custom_props": {k: v for k, v in pbone.items() if not k.startswith("_") and isinstance(v, (int, float, str))}
        }
        
        rig_info["bones"][bone.name] = bone_data

    output_path = os.path.join(os.path.dirname(bpy.data.filepath), "rig_info.json")
    
    if not bpy.data.filepath:
        output_path = os.path.join(os.path.expanduser("~"), "rig_info.json")

    with open(output_path, 'w') as f:
        json.dump(rig_info, f, indent=2)

    print(f"✅ Información del Rig guardada en: {output_path}")

if __name__ == "__main__":
    analyze_rig()