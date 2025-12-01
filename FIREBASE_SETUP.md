# 🔥 Configuración de Firebase para Motion Capture

Este documento explica cómo configurar Firebase para guardar tus grabaciones de motion capture en la nube.

## 📋 Requisitos Previos

1. Una cuenta de Google
2. Un proyecto de Firebase creado
3. Python 3.7 o superior
4. Dependencias instaladas: `pip install -r requirements.txt`

---

## 🚀 Opción 1: Service Account Key (Recomendado para desarrollo)

### Paso 1: Crear un Proyecto en Firebase

1. Ve a [Firebase Console](https://console.firebase.google.com/)
2. Haz clic en "Agregar proyecto" o selecciona un proyecto existente
3. Sigue los pasos de configuración

### Paso 2: Habilitar Firestore

1. En el menú lateral, ve a **Firestore Database**
2. Haz clic en **"Crear base de datos"**
3. Selecciona **modo de prueba** (o producción si prefieres configurar reglas)
4. Elige la región más cercana (por ejemplo: `us-central1`)

### Paso 3: Generar la Clave de Cuenta de Servicio

1. Ve a **Configuración del proyecto** (⚙️ en la esquina superior izquierda)
2. Selecciona la pestaña **"Cuentas de servicio"**
3. Haz clic en **"Generar nueva clave privada"**
4. Confirma y descarga el archivo JSON

### Paso 4: Configurar en tu Proyecto

1. Renombra el archivo descargado a `serviceAccountKey.json`
2. Colócalo en la **raíz de tu proyecto** (mismo nivel que `run_mocap.py`)

```
blender_v2/
├── serviceAccountKey.json  ← Aquí
├── run_mocap.py
├── requirements.txt
└── src/
```

3. **¡IMPORTANTE!** Agrega este archivo a `.gitignore` para no subirlo a GitHub:

```bash
echo "serviceAccountKey.json" >> .gitignore
```

---

## 🌐 Opción 2: Credenciales por Defecto (gcloud)

Si ya tienes `gcloud` instalado y configurado:

```bash
# Iniciar sesión
gcloud auth application-default login

# Configurar el proyecto
gcloud config set project TU_PROJECT_ID
```

El sistema usará automáticamente estas credenciales.

---

## ✅ Verificar la Configuración

Ejecuta este script de prueba:

```python
# test_firebase.py
from src.firebase_config import is_firebase_available, upload_data

if is_firebase_available():
    print("✓ Firebase configurado correctamente")
    
    # Prueba de subida
    test_data = {
        "test": True,
        "message": "Prueba de conexión"
    }
    success, doc_id = upload_data(test_data, collection_name="test")
    
    if success:
        print(f"✓ Datos de prueba subidos exitosamente (ID: {doc_id})")
    else:
        print("✗ Error al subir datos")
else:
    print("✗ Firebase no está configurado")
```

---

## 📊 Ver tus Datos en Firebase

1. Ve a [Firebase Console](https://console.firebase.google.com/)
2. Selecciona tu proyecto
3. Ve a **Firestore Database** en el menú lateral
4. Verás tus grabaciones en la colección `mocap_recordings`

Cada documento contendrá:
- `recording_id`: Identificador único
- `framerate`: FPS de la grabación
- `total_frames`: Número de frames
- `duration`: Duración en segundos
- `frames`: Array con todos los datos de pose
- `uploaded_at`: Timestamp de subida

---

## 🔧 Solución de Problemas

### "Firebase no disponible"

**Causa:** No se encontró `serviceAccountKey.json` ni credenciales por defecto.

**Solución:** Sigue los pasos de la Opción 1 o 2 arriba.

### "Permission denied" en Firestore

**Causa:** Las reglas de seguridad de Firestore bloquean la escritura.

**Solución:** En Firebase Console → Firestore → Reglas, usa esto para desarrollo:

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /{document=**} {
      allow read, write: if true;  // ⚠️ Solo para desarrollo
    }
  }
}
```

⚠️ **Para producción**, configura reglas más restrictivas.

### "Failed to initialize Firebase"

**Causa:** Error en el archivo JSON o permisos.

**Solución:**
1. Verifica que el archivo `serviceAccountKey.json` sea válido JSON
2. Asegúrate de que esté en la raíz del proyecto
3. Verifica que Firestore esté habilitado en tu proyecto

---

## 🔒 Seguridad

### ⚠️ NUNCA subas `serviceAccountKey.json` a GitHub

Este archivo contiene credenciales sensibles. Siempre:

1. Agrégalo a `.gitignore`
2. No lo compartas públicamente
3. Revócalo si accidentalmente lo expones

### Revocar una clave comprometida

1. Ve a Firebase Console → Configuración → Cuentas de servicio
2. Elimina la cuenta de servicio comprometida
3. Genera una nueva clave

---

## 💡 Consejos

1. **Modo offline:** Si Firebase no está disponible, el sistema **automáticamente guarda solo localmente** en `output_animation.json`

2. **Ver estadísticas:** En Firebase Console puedes ver cuánto almacenamiento usas (plan gratuito: 1 GB)

3. **Limpiar datos antiguos:** Crea un script para eliminar grabaciones antiguas:

```python
from firebase_admin import firestore
import firebase_admin

# Inicializar
if not firebase_admin._apps:
    firebase_admin.initialize_app()

db = firestore.client()

# Eliminar documentos de prueba
test_docs = db.collection('test').stream()
for doc in test_docs:
    doc.reference.delete()
    print(f"Eliminado: {doc.id}")
```

---

## 📞 Soporte

Si tienes problemas:

1. Verifica que `firebase-admin` esté instalado: `pip install firebase-admin`
2. Revisa los logs de error en la consola
3. Consulta la [documentación oficial de Firebase](https://firebase.google.com/docs/admin/setup)
