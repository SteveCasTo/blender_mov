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

def test_hierarchy():
    solver = PoseSolver()
    
    # Create 33 dummy landmarks
    landmarks = [MockLandmark(0.0, 0.0, 0.0) for _ in range(33)]
    
    # Setup a pose where Arm is rotated 90 deg, but Forearm is straight relative to Arm.
    # If logic is correct:
    # Upper Arm Global: 90 deg
    # Forearm Global: 90 deg
    # -> Forearm Local: 0 deg (Identity)
    
    # T-Pose (Rest): Left Arm points +X
    # Rotated Pose: Left Arm points +Y (Forward in Blender, Down in MP?)
    # Let's use Blender coords for mental model, then map to MP.
    # Blender: Arm +X -> Arm +Y (90 deg Z rotation)
    # MP: Arm +X -> Arm +Z (Depth)
    
    # Shoulder (11) at Origin
    landmarks[11] = MockLandmark(0.0, 0.0, 0.0)
    # Elbow (13) at (0, 0, 0.5) -> +Z depth (Forward in Blender)
    landmarks[13] = MockLandmark(0.0, 0.0, 0.5)
    # Wrist (15) at (0, 0, 1.0) -> +Z depth
    landmarks[15] = MockLandmark(0.0, 0.0, 1.0)
    
    print("--- Testing Hierarchy ---")
    bones = solver.solve(landmarks)
    
    if "upper_arm.L" in bones and "forearm.L" in bones:
        ua = np.array(bones["upper_arm.L"])
        fa = np.array(bones["forearm.L"])
        
        print(f"Upper Arm Local: {ua}")
        print(f"Forearm Local: {fa}")
        
        # Forearm should be close to Identity [1, 0, 0, 0] because it's straight relative to Upper Arm
        # Upper Arm should be rotated ~90 deg
        
        if np.abs(fa[0] - 1.0) < 0.1: # w close to 1
            print("SUCCESS: Forearm Local is Identity (Correct Hierarchy)")
        else:
            print("FAILURE: Forearm Local is NOT Identity (Double Rotation?)")
            
    else:
        print("MISSING bones")

if __name__ == "__main__":
    test_hierarchy()
