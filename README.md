# Motion Capture System - Project Documentation

This project implements a real-time motion capture system using MediaPipe and Python, with data transmission to Blender and storage in Firebase.

## File Structure & Explanations

### Root Directory

- **`run_mocap.py`**: The main entry point of the application. It initializes the webcam, runs the pose detection loop, solves for bone rotations, sends data to Blender, and handles recording.
- **`blender_receiver.py`**: A script intended to be run _inside_ Blender. It listens for data sent by `run_mocap.py` and updates the 3D character's armature in real-time.
- **`requirements.txt`**: Lists the Python dependencies required to run the project (e.g., `mediapipe`, `opencv-python`, `firebase-admin`).
- **`output_animation.json`**: The default output file where recorded motion data is saved locally.

### `src/` Directory

- **`src/__init__.py`**: Makes the `src` directory a Python package, allowing imports from it.
- **`src/capture.py`**: Contains the `PoseDetector` class. It handles the webcam input and uses MediaPipe to detect pose landmarks.
- **`src/solver.py`**: Contains the `PoseSolver` class. This is the core mathematical engine that converts raw 3D landmarks (points in space) into bone rotations (quaternions) suitable for a 3D skeleton.
- **`src/network.py`**: Contains the `MocapSender` class. It handles sending the processed bone data to Blender via UDP sockets.
- **`src/exporter.py`**: Contains the `MocapRecorder` class. It manages recording the motion data frame-by-frame and saving it to a JSON file and Firebase.
- **`src/firebase_config.py`**: Handles the connection to Firebase. It initializes the app using credentials and provides the function to upload data.

### `tests/` Directory

- **`tests/test_math.py`**: Contains unit tests to verify the correctness of the mathematical calculations (e.g., vector operations, quaternion conversions).

## Setup for Firebase

Para habilitar el guardado en Firebase:

1. **Lee la guía completa:** Consulta [`FIREBASE_SETUP.md`](FIREBASE_SETUP.md) para instrucciones detalladas
2. **Descarga tu clave:** Obtén `serviceAccountKey.json` desde Firebase Console
3. **Colócala en la raíz:** Guarda el archivo en el mismo directorio que `run_mocap.py`
4. **Prueba la conexión:** Ejecuta `python test_firebase_connection.py`

### Modo Sin Firebase

El sistema funciona **perfectamente sin Firebase**. Si no lo configuras:
- ✅ Las grabaciones se guardan localmente en `output_animation.json`
- ⚠️ Verás advertencias sobre Firebase no disponible (son normales)
- ✅ Todas las demás funciones siguen funcionando

### Verificar Estado de Firebase

```python
from src.firebase_config import is_firebase_available

if is_firebase_available():
    print("Firebase configurado ✓")
else:
    print("Solo guardado local ⚠")
```