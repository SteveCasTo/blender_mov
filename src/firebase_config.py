import firebase_admin
from firebase_admin import credentials, firestore, db
import os
import time

# Variable global para rastrear si Firebase está inicializado
_firebase_initialized = False
_firebase_available = False

def initialize_firebase():
    """
    Inicializa Firebase Admin SDK.
    
    Opciones de autenticación (en orden de prioridad):
    1. serviceAccountKey.json en el directorio raíz
    2. Variable de entorno GOOGLE_APPLICATION_CREDENTIALS
    3. Credenciales por defecto de gcloud (si estás logueado)
    
    Returns:
        bool: True si la inicialización fue exitosa, False en caso contrario
    """
    global _firebase_initialized, _firebase_available
    
    if _firebase_initialized:
        return _firebase_available
    
    _firebase_initialized = True
    
    try:
        # Si ya existe una app inicializada, la usamos
        if firebase_admin._apps:
            print("✓ Firebase ya estaba inicializado.")
            _firebase_available = True
            return True
        
        # Opción 1: Buscar serviceAccountKey.json en el directorio raíz
        cred_path = "serviceAccountKey.json"
        
        if os.path.exists(cred_path):
            print(f"✓ Encontrado {cred_path}, usando para autenticación Firebase.")
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            _firebase_available = True
            return True
        
        # Opción 2: Variable de entorno GOOGLE_APPLICATION_CREDENTIALS
        if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            print("✓ Usando GOOGLE_APPLICATION_CREDENTIALS para Firebase.")
            cred = credentials.ApplicationDefault()
            firebase_admin.initialize_app(cred)
            _firebase_available = True
            return True
        
        # Opción 3: Credenciales por defecto
        print("⚠ No se encontró serviceAccountKey.json")
        print("  Intentando usar credenciales por defecto de gcloud...")
        
        firebase_admin.initialize_app()
        print("✓ Firebase inicializado con credenciales por defecto.")
        _firebase_available = True
        return True
        
    except Exception as e:
        print(f"✗ Error al inicializar Firebase: {e}")
        print("\n📋 Para configurar Firebase:")
        print("   1. Ve a Firebase Console: https://console.firebase.google.com/")
        print("   2. Selecciona tu proyecto")
        print("   3. Ve a Configuración del proyecto → Cuentas de servicio")
        print("   4. Genera una nueva clave privada (JSON)")
        print("   5. Guarda el archivo como 'serviceAccountKey.json' en la raíz del proyecto")
        print("   6. Asegúrate de que Firestore esté habilitado en tu proyecto\n")
        _firebase_available = False
        return False

def upload_data(data, collection_name="mocap_recordings"):
    """
    Sube datos de captura de movimiento a Firebase Firestore.
    
    Args:
        data (dict): Los datos a guardar (debe incluir frames, framerate, etc.)
        collection_name (str): Nombre de la colección en Firestore
    
    Returns:
        tuple: (success: bool, doc_id: str or None)
    """
    try:
        # Intentar inicializar Firebase
        if not initialize_firebase():
            print("⚠ Firebase no disponible. Datos NO subidos a la nube.")
            return (False, None)
        
        # Obtener cliente de Firestore
        db = firestore.client()
        
        # Crear nombre único para el documento
        timestamp = int(time.time())
        doc_name = f"recording_{timestamp}"
        
        # Agregar metadata adicional
        data_with_metadata = {
            **data,
            "uploaded_at": firestore.SERVER_TIMESTAMP,
            "recording_id": doc_name
        }
        
        # Subir a Firestore
        doc_ref = db.collection(collection_name).document(doc_name)
        doc_ref.set(data_with_metadata)
        
        print(f"✓ Datos guardados exitosamente en Firebase Firestore")
        print(f"  Colección: {collection_name}")
        print(f"  Documento: {doc_name}")
        print(f"  Total frames: {data.get('total_frames', 'N/A')}")
        print(f"  Duración: {data.get('duration', 'N/A')}s")
        
        return (True, doc_name)
        
    except Exception as e:
        print(f"✗ Error al subir datos a Firebase: {e}")
        print("  Los datos locales se guardaron correctamente en JSON.")
        return (False, None)

def is_firebase_available():
    """
    Verifica si Firebase está disponible y configurado.
    
    Returns:
        bool: True si Firebase está disponible
    """
    global _firebase_initialized, _firebase_available
    
    if not _firebase_initialized:
        initialize_firebase()
    
    return _firebase_available