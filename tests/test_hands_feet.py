import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from solver import PoseSolver

class MockLandmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

def test_solver():
    solver = PoseSolver()
    
    # Create 33 dummy landmarks
    landmarks = [MockLandmark(0.0, 0.0, 0.0) for _ in range(33)]
    
    # Set specific landmarks for Left Hand to test orientation
    # T-Pose-ish: Arm out to Left (+X in Blender -> +X in MP?)
    # Wait, MP coords: +X is Left (on screen), +Y is Down.
    # Let's just set them to form a valid triangle.
    
    # Wrist (15) at origin (relative)
    landmarks[15] = MockLandmark(0.5, 0.5, 0.0)
    # Index (19) at +0.1 X
    landmarks[19] = MockLandmark(0.6, 0.5, 0.0)
    # Pinky (17) at +0.1 X, +0.05 Y (Down)
    landmarks[17] = MockLandmark(0.6, 0.55, 0.0)
    
    # Feet
    # Ankle (27)
    landmarks[27] = MockLandmark(0.2, 0.8, 0.0)
    # Heel (29)
    landmarks[29] = MockLandmark(0.2, 0.85, 0.0)
    # Toe (31)
    landmarks[31] = MockLandmark(0.2, 0.9, -0.1) # Forward (Z is depth in MP)

    try:
        bones = solver.solve(landmarks)
        print("Solver ran successfully.")
        
        required_bones = ["hand.L", "hand.R", "foot.L", "foot.R"]
        for bone in required_bones:
            if bone in bones:
                print(f"{bone}: {bones[bone]}")
                # Check if quaternion is valid (norm close to 1)
                q = np.array(bones[bone])
                if np.abs(np.linalg.norm(q) - 1.0) < 1e-6:
                    print(f"  -> Valid Quaternion")
                else:
                    print(f"  -> INVALID Quaternion norm: {np.linalg.norm(q)}")
            else:
                print(f"MISSING {bone}")
                
    except Exception as e:
        print(f"Solver failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_solver()
