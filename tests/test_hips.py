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

def test_hips():
    solver = PoseSolver()
    
    # Create 33 dummy landmarks
    landmarks = [MockLandmark(0.0, 0.0, 0.0) for _ in range(33)]
    
    # Initial Pose (Center)
    # Hips at (0.5, 0.5, 0.0)
    landmarks[23] = MockLandmark(0.4, 0.5, 0.0) # L Hip
    landmarks[24] = MockLandmark(0.6, 0.5, 0.0) # R Hip
    
    print("--- Frame 1 (Initial) ---")
    bones = solver.solve(landmarks)
    if "hips" in bones:
        print(f"Hips: {bones['hips']}")
    else:
        print("MISSING hips")
        
    # Frame 2 (Moved Right +0.1 in X)
    # MP X+ is Right? Let's assume.
    landmarks[23] = MockLandmark(0.5, 0.5, 0.0)
    landmarks[24] = MockLandmark(0.7, 0.5, 0.0)
    
    print("\n--- Frame 2 (Moved Right) ---")
    bones = solver.solve(landmarks)
    if "hips" in bones:
        print(f"Hips: {bones['hips']}")
        # Expecting X translation
    else:
        print("MISSING hips")

if __name__ == "__main__":
    test_hips()
