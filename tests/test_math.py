import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.solver import PoseSolver

def test_rotation():
    solver = PoseSolver()
    
    # Test 1: 90 degree rotation around Z (X -> Y)
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    
    q = solver.rotation_between_vectors(v1, v2)
    print(f"Test 1 (X -> Y): {q}")
    
    # Expected: [0.707, 0, 0, 0.707] (w, x, y, z)
    expected = np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])
    if np.allclose(q, expected, atol=0.01) or np.allclose(q, -expected, atol=0.01):
        print("PASS")
    else:
        print(f"FAIL. Expected {expected}")

    # Test 2: No rotation
    v3 = np.array([0, 0, 1])
    q2 = solver.rotation_between_vectors(v3, v3)
    print(f"Test 2 (Identity): {q2}")
    if np.allclose(q2, [1, 0, 0, 0], atol=0.01):
        print("PASS")
    else:
        print("FAIL")

if __name__ == "__main__":
    test_rotation()
