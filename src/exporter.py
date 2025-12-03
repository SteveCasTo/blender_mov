import json
import time
from src.firebase_config import upload_data, is_firebase_available

class MocapRecorder:
    def __init__(self, output_file="animation.json"):
        self.output_file = output_file
        self.frames = []
        self.start_time = None
        self.framerate = 30
        
        # Verificar Firebase al iniciar
        if is_firebase_available():
            print("✓ Firebase está configurado y listo")
        else:
            print("⚠ Firebase no disponible - solo se guardará localmente")

    def start(self):
        self.start_time = time.time()
        self.frames = []

    def record_frame(self, bone_data):
        if self.start_time is None:
            self.start()
        
        timestamp = time.time() - self.start_time
        frame = {
            "timestamp": round(timestamp, 4),
            "bones": bone_data
        }
        self.frames.append(frame)

    def save(self):
        if not self.frames:
            print("⚠ No hay frames para guardar")
            return
        
        data = {
            "framerate": self.framerate,
            "total_frames": len(self.frames),
            "duration": round(self.frames[-1]["timestamp"], 4) if self.frames else 0,
            "frames": self.frames
        }
        
        # Guardar localmente
        try:
            with open(self.output_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✓ Animación guardada localmente: {self.output_file}")
            print(f"  Frames: {data['total_frames']}, Duración: {data['duration']}s")
        except Exception as e:
            print(f"✗ Error guardando archivo local: {e}")
            return
        
        # Intentar subir a Firebase
        success, doc_id = upload_data(data)
        
        if success:
            print(f"✓ Animación sincronizada con Firebase (ID: {doc_id})")
        else:
            print("⚠ No se pudo subir a Firebase, pero el archivo local está guardado")