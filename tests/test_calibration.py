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

def test_calibration():
    solver = PoseSolver()
    
    # Create 33 dummy landmarks representing a T-Pose
    landmarks = [MockLandmark(0.0, 0.0, 0.0) for _ in range(33)]
    
    # Setup a T-Pose
    # Spine: Hip(0,0,0) -> Shoulder(0,0,1)
    landmarks[23] = MockLandmark(0.0, 0.0, 0.0) # L Hip
    landmarks[24] = MockLandmark(0.0, 0.0, 0.0) # R Hip
    landmarks[11] = MockLandmark(0.0, 0.0, 1.0) # L Shoulder
    landmarks[12] = MockLandmark(0.0, 0.0, 1.0) # R Shoulder
    
    # Left Arm: Shoulder(0,0,1) -> Elbow(1,0,1) -> Wrist(2,0,1)
    landmarks[13] = MockLandmark(1.0, 0.0, 1.0) # L Elbow
    landmarks[15] = MockLandmark(2.0, 0.0, 1.0) # L Wrist
    
    print("--- Testing Calibration ---")
    
    # 1. Calibrate
    print("Calibrating...")
    solver.calibrate(landmarks)
    
    # 2. Solve same pose
    print("Solving same pose...")
    bones = solver.solve(landmarks)
    
    # 3. Check Rotations
    # Should be Identity [1, 0, 0, 0]
    
    if "upper_arm.L" in bones:
        q = np.array(bones["upper_arm.L"])
        print(f"Upper Arm L: {q}")
        
        # Check if close to Identity
        if np.allclose(q, [1, 0, 0, 0], atol=0.1):
            print("SUCCESS: Rotation is Identity after calibration.")
        else:
            print("FAILURE: Rotation is NOT Identity.")
            
    if "spine" in bones:
        q = np.array(bones["spine"])
        print(f"Spine: {q}")
        if np.allclose(q, [1, 0, 0, 0], atol=0.1):
            print("SUCCESS: Spine is Identity.")

if __name__ == "__main__":
    test_calibration()
