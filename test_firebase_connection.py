import sys
from src.firebase_config import is_firebase_available, upload_data

def test_firebase_connection():
    print("="*60)
    print("🔥 PRUEBA DE CONFIGURACIÓN DE FIREBASE")
    print("="*60)
    print()
    
    # 1. Verificar disponibilidad
    print("1️⃣  Verificando disponibilidad de Firebase...")
    if is_firebase_available():
        print("   ✓ Firebase está configurado y disponible")
    else:
        print("   ✗ Firebase NO está configurado")
        print()
        print("📋 Pasos para configurar Firebase:")
        print("   1. Lee el archivo FIREBASE_SETUP.md")
        print("   2. Descarga tu serviceAccountKey.json")
        print("   3. Colócalo en la raíz del proyecto")
        print()
        return False
    
    print()
    
    # 2. Probar subida de datos
    print("2️⃣  Probando subida de datos a Firestore...")
    test_data = {
        "test": True,
        "message": "Prueba de conexión desde test_firebase_connection.py",
        "framerate": 30,
        "total_frames": 1,
        "duration": 0.033,
        "frames": [
            {
                "timestamp": 0.0,
                "bones": {
                    "spine": [1.0, 0.0, 0.0, 0.0]
                }
            }
        ]
    }
    
    success, doc_id = upload_data(test_data, collection_name="test_connection")
    
    if success:
        print(f"   ✓ Datos de prueba subidos exitosamente")
        print(f"   📄 ID del documento: {doc_id}")
        print()
        print("🎉 ¡Todo funciona correctamente!")
        print()
        print("🔍 Verifica tus datos en Firebase Console:")
        print("   https://console.firebase.google.com/")
        print("   → Firestore Database → Colección: 'test_connection'")
        print()
        return True
    else:
        print("   ✗ Error al subir datos de prueba")
        print("   Revisa los mensajes de error arriba")
        print()
        return False

if __name__ == "__main__":
    success = test_firebase_connection()
    sys.exit(0 if success else 1)
