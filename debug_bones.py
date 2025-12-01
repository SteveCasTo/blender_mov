import bpy

# Select your Armature in the 3D View first!
obj = bpy.context.object

if obj and obj.type == 'ARMATURE':
    print("-" * 30)
    print(f"Bone Names for Armature: {obj.name}")
    print("-" * 30)
    for bone in obj.pose.bones:
        print(f'"{bone.name}"')
    print("-" * 30)
    print("COPY THE ABOVE LIST AND SHARE IT")
else:
    print("ERROR: Please select an Armature object first.")
