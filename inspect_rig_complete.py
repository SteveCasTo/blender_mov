"""
Script DEEP SCAN para inspeccionar el rig de Blender
Analiza TODOS los huesos, jerarquías, constraints y propiedades sin filtros.
Ejecuta esto en Blender (Text Editor > New > Pega el código > Run Script)
"""

import bpy
import mathutils

def analyze_bone_recursive(pbone, f, indent=0):
    """Analiza un hueso y sus hijos recursivamente"""
    prefix = "  " * indent
    bone = pbone.bone
    
    # --- Encabezado del Hueso ---
    f.write(f"\n{prefix}► {pbone.name}\n")
    f.write(f"{prefix}  Parent: {pbone.parent.name if pbone.parent else 'None'}\n")
    
    # --- Transformaciones (World Space) ---
    # Usamos matrix.translation para la posición real en world space
    world_pos = pbone.matrix.translation
    f.write(f"{prefix}  Head (World): ({world_pos.x:.4f}, {world_pos.y:.4f}, {world_pos.z:.4f})\n")
    
    # Rotación
    if pbone.rotation_mode == 'QUATERNION':
        rot = pbone.rotation_quaternion
        f.write(f"{prefix}  Rot (Quat): ({rot.w:.3f}, {rot.x:.3f}, {rot.y:.3f}, {rot.z:.3f})\n")
    elif pbone.rotation_mode == 'AXIS_ANGLE':
        rot = pbone.rotation_axis_angle
        f.write(f"{prefix}  Rot (Axis): ({rot[0]:.3f}, {rot[1]:.3f}, {rot[2]:.3f}, {rot[3]:.3f})\n")
    else:
        rot = pbone.rotation_euler
        f.write(f"{prefix}  Rot (Euler {pbone.rotation_mode}): ({rot.x:.3f}, {rot.y:.3f}, {rot.z:.3f})\n")

    # --- Constraints ---
    if pbone.constraints:
        f.write(f"{prefix}  Constraints:\n")
        for c in pbone.constraints:
            target = getattr(c, 'target', None)
            subtarget = getattr(c, 'subtarget', '')
            target_name = target.name if target else "None"
            f.write(f"{prefix}    • {c.name} ({c.type}) -> Target: {target_name} [{subtarget}] (Inf: {c.influence:.2f})\n")
            # Detalles extra para IK
            if c.type == 'IK':
                f.write(f"{prefix}      - Pole: {getattr(c, 'pole_target', None) and c.pole_target.name} [{getattr(c, 'pole_subtarget', '')}]\n")
                f.write(f"{prefix}      - Chain Len: {c.chain_count}\n")

    # --- Custom Properties ---
    props = [k for k in pbone.keys() if not k.startswith('_') and k not in ['cycles', 'path_resolve']]
    if props:
        f.write(f"{prefix}  Properties:\n")
        for k in props:
            val = pbone[k]
            if isinstance(val, float):
                f.write(f"{prefix}    - {k}: {val:.3f}\n")
            else:
                f.write(f"{prefix}    - {k}: {val}\n")

    # --- Matriz (Solo ejes principales para no saturar) ---
    # X (Right), Y (Forward/Tail), Z (Up)
    mat = pbone.matrix
    f.write(f"{prefix}  Axis Orientation (World):\n")
    f.write(f"{prefix}    X: ({mat[0][0]:.2f}, {mat[0][1]:.2f}, {mat[0][2]:.2f})\n")
    f.write(f"{prefix}    Y: ({mat[1][0]:.2f}, {mat[1][1]:.2f}, {mat[1][2]:.2f})\n")
    f.write(f"{prefix}    Z: ({mat[2][0]:.2f}, {mat[2][1]:.2f}, {mat[2][2]:.2f})\n")

    # --- Recursión Hijos ---
    # Buscar hijos en pose bones
    children = [child for child in pbone.children]
    # Ordenar por nombre para consistencia
    children.sort(key=lambda x: x.name)
    
    for child in children:
        analyze_bone_recursive(child, f, indent + 1)

def inspect_deep_scan():
    """Inspección PROFUNDA del rig"""
    
    obj = bpy.context.active_object
    
    if not obj or obj.type != 'ARMATURE':
        print("ERROR: Selecciona un armature primero")
        return
    
    output_file = "C:/Users/Steven/.gemini/rig_deep_analysis.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write(f"DEEP SCAN RIG ANALYSIS: {obj.name}\n")
        f.write("="*100 + "\n\n")
        
        # 1. General Info
        f.write(f"Location: {obj.location}\n")
        f.write(f"Rotation: {obj.rotation_euler}\n")
        f.write(f"Scale: {obj.scale}\n\n")
        
        # 2. Hierarchy Scan
        f.write("="*100 + "\n")
        f.write("HIERARCHY & FULL BONE DETAILS\n")
        f.write("="*100 + "\n")
        
        # Encontrar huesos raiz (sin padre)
        root_bones = [b for b in obj.pose.bones if not b.parent]
        root_bones.sort(key=lambda x: x.name)
        
        for root in root_bones:
            analyze_bone_recursive(root, f)
            f.write("\n" + "-"*50 + "\n")

    print(f"\n{'='*100}")
    print(f"ANÁLISIS DEEP SCAN GUARDADO EN: {output_file}")
    print(f"{'='*100}")

if __name__ == "__main__":
    inspect_deep_scan()
