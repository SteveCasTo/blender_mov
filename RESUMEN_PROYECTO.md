# 📊 RESUMEN EJECUTIVO DEL PROYECTO - Motion Capture System

## 🎯 ¿Qué hace este proyecto?

Sistema de **captura de movimiento en tiempo real** que:
1. Detecta tu cuerpo con la webcam usando IA (MediaPipe)
2. Convierte la pose detectada en rotaciones 3D de huesos
3. Envía los datos a Blender para animar un personaje en tiempo real
4. Graba las sesiones en JSON y las sube a Firebase (opcional)

---

## 📁 Estructura del Proyecto

```
blender_v2/
│
├── 🎬 SCRIPTS PRINCIPALES
│   ├── run_mocap.py              # ← Ejecutar ESTO para iniciar el sistema
│   ├── blender_receiver.py       # ← Ejecutar DENTRO de Blender para recibir datos
│   └── test_firebase_connection.py  # ← Probar configuración de Firebase
│
├── 📦 MÓDULOS (src/)
│   ├── capture.py                # Captura de webcam + MediaPipe
│   ├── solver.py                 # Matemática: landmarks → rotaciones
│   ├── network.py                # Envío UDP a Blender
│   ├── exporter.py               # Grabación y guardado
│   └── firebase_config.py        # Configuración de Firebase
│
├── 📄 DOCUMENTACIÓN
│   ├── README.md                 # Documentación general
│   ├── FIREBASE_SETUP.md         # Guía de configuración de Firebase
│   └── RESUMEN_PROYECTO.md       # Este archivo
│
├── ⚙️ CONFIGURACIÓN
│   ├── requirements.txt          # Dependencias Python
│   ├── .gitignore               # Archivos a ignorar en Git
│   └── serviceAccountKey.json   # 🔒 TU CLAVE DE FIREBASE (no incluido)
│
└── 📊 DATOS
    └── output_animation.json     # Grabaciones guardadas localmente
```

---

## 🔄 Flujo de Funcionamiento

```
┌─────────────┐
│   WEBCAM    │
└──────┬──────┘
       │ Video frames
       ↓
┌─────────────────────┐
│  MediaPipe Pose     │  ← Detecta 33 puntos del cuerpo
│  (capture.py)       │
└──────┬──────────────┘
       │ 3D Landmarks (x, y, z)
       ↓
┌─────────────────────┐
│  PoseSolver         │  ← Calcula rotaciones de huesos
│  (solver.py)        │     usando cuaterniones
└──────┬──────────────┘
       │ Bone rotations (quaternions)
       ↓
   ┌───┴────┐
   │        │
   ↓        ↓
┌────────┐  ┌────────────┐
│ BLENDER│  │  RECORDER  │
│ (UDP)  │  │ (exporter) │
└────────┘  └──────┬─────┘
                   │
              ┌────┴─────┐
              │          │
              ↓          ↓
         ┌────────┐  ┌──────────┐
         │ JSON   │  │ FIREBASE │
         │ Local  │  │ (cloud)  │
         └────────┘  └──────────┘
```

---

## 🚀 Cómo Usar el Sistema

### 1️⃣ Instalación (Primera vez)

```bash
# Instalar dependencias
pip install -r requirements.txt
```

### 2️⃣ Configurar Firebase (Opcional)

```bash
# Seguir la guía
# Ver: FIREBASE_SETUP.md

# Probar conexión
python test_firebase_connection.py
```

### 3️⃣ Ejecutar el Sistema

**Terminal 1 - Captura de movimiento:**
```bash
python run_mocap.py
```

**Terminal 2 - Blender (opcional):**
1. Abrir Blender
2. Cargar un personaje con armature
3. Scripting tab → Abrir `blender_receiver.py`
4. Ejecutar el script

### 4️⃣ Controles

- **R** = Iniciar/Detener grabación
- **Q** = Salir
- **ESC** (en Blender) = Detener receptor

---

## 📚 Detalles Técnicos

### 🧮 Matemática Clave (solver.py)

El `PoseSolver` realiza la conversión más importante:

1. **Entrada:** 33 landmarks 3D de MediaPipe
   - Formato: `(x, y, z)` en coordenadas de cámara

2. **Conversión de coordenadas:**
   ```
   MediaPipe        →  Blender
   X (izq-der)      →  X (izq-der)
   Y (arriba-abajo) →  Z (arriba-abajo) 
   Z (profundidad)  →  Y (adelante-atrás)
   ```

3. **Cálculo de vectores de huesos:**
   - Ejemplo: `brazo = hombro → codo`

4. **Rotación entre vectores:**
   - T-pose (reposo) → Pose actual
   - Método: Cuaterniones (w, x, y, z)

5. **Salida:** Diccionario de rotaciones por hueso
   ```python
   {
     "spine": [0.98, 0.16, 0.00, 0.00],
     "upper_arm.L": [0.79, 0.00, 0.60, -0.01],
     ...
   }
   ```

### 🎨 Huesos Soportados

| Hueso           | Descripción          | Landmarks      |
|-----------------|----------------------|----------------|
| `spine`         | Columna vertebral    | cadera→hombros |
| `neck`          | Cuello               | hombros→nariz  |
| `upper_arm.L/R` | Brazo superior       | hombro→codo    |
| `forearm.L/R`   | Antebrazo            | codo→muñeca    |
| `thigh.L/R`     | Muslo                | cadera→rodilla |
| `shin.L/R`      | Espinilla            | rodilla→tobillo|

### 🌐 Comunicación en Red

**Protocolo:** UDP Socket
- **Puerto:** 9000
- **Host:** 127.0.0.1 (localhost)
- **Formato:** JSON serializado

**Ventajas de UDP:**
- Baja latencia (ideal para tiempo real)
- No espera confirmación
- Si un paquete se pierde, no importa (llegará el siguiente)

---

## 🔥 Firebase - Detalles

### Estructura de Datos en Firestore

**Colección:** `mocap_recordings`

**Documento ejemplo:**
```json
{
  "recording_id": "recording_1700000000",
  "framerate": 30,
  "total_frames": 494,
  "duration": 41.88,
  "uploaded_at": "2025-11-22T10:30:00Z",
  "frames": [
    {
      "timestamp": 0.0622,
      "bones": {
        "spine": [0.987, 0.160, 0.001, 0.000],
        "upper_arm.L": [...],
        ...
      }
    },
    ...
  ]
}
```

### Modos de Autenticación

| Método | Archivo | Uso |
|--------|---------|-----|
| Service Account | `serviceAccountKey.json` | Desarrollo local |
| Application Default | Variable de entorno | Producción |
| gcloud | Login de usuario | Desarrollo rápido |

---

## ✅ Estado Actual y Cambios Realizados

### ✨ Mejoras Implementadas

1. **Firebase completamente funcional:**
   - ✅ Soporte para múltiples métodos de autenticación
   - ✅ Manejo robusto de errores
   - ✅ Modo offline automático (guarda local si Firebase falla)
   - ✅ Mensajes claros y útiles

2. **Hueso del cuello agregado:**
   - ✅ Cálculo de rotación del cuello (mid-shoulder → nariz)
   - ✅ Ahora se exporta en los datos

3. **Documentación mejorada:**
   - ✅ Guía completa de Firebase (`FIREBASE_SETUP.md`)
   - ✅ Script de prueba (`test_firebase_connection.py`)
   - ✅ README actualizado
   - ✅ `.gitignore` para proteger credenciales

4. **Mejor feedback al usuario:**
   - ✅ Mensajes con iconos (✓, ✗, ⚠)
   - ✅ Información de estado de Firebase al iniciar
   - ✅ Detalles de uploads (ID, frames, duración)

### 🐛 Problemas Resueltos

| # | Problema | Solución |
|---|----------|----------|
| 1 | Firebase no configurado | Múltiples métodos de auth + docs |
| 2 | URL hardcodeada | Uso de Firestore (no necesita URL) |
| 3 | Cuello sin calcular | Agregado cálculo neck |
| 4 | Sin protección de credenciales | `.gitignore` creado |
| 5 | Errores confusos | Mensajes claros con emojis |

### 📝 Notas Importantes

1. **MediaPipe Landmarks:** El sistema usa el modelo `pose_landmarks` de MediaPipe que detecta 33 puntos clave del cuerpo.

2. **Coordinate System:** Blender usa Z-up, MediaPipe usa Y-down. La conversión está en `solver.py`.

3. **Quaternion Order:** `[w, x, y, z]` - el componente escalar (w) va primero.

4. **Bone Mapping:** Los nombres en `blender_receiver.py` deben coincidir EXACTAMENTE con tu rig de Blender.

---

## 🎯 Próximos Pasos Sugeridos

### Para mejorar el proyecto:

1. **Más huesos:**
   - [ ] Manos (dedos)
   - [ ] Pies (dedos)
   - [ ] Cabeza (rotación independiente)

2. **Calibración:**
   - [ ] T-pose automática al iniciar
   - [ ] Ajuste de proporción del esqueleto

3. **Filtrado:**
   - [ ] Suavizado de movimientos (filtro de Kalman)
   - [ ] Reducción de jitter

4. **UI/UX:**
   - [ ] Interfaz gráfica simple
   - [ ] Visualización de esqueleto 2D
   - [ ] Configuración de parámetros en tiempo real

5. **Exportación:**
   - [ ] Formato BVH (estándar de mocap)
   - [ ] Integración directa con Blender Action Editor

---

## 🛠️ Solución de Problemas Comunes

### "ModuleNotFoundError: No module named 'mediapipe'"
```bash
pip install mediapipe opencv-python numpy scipy firebase-admin
```

### "No se detecta la pose"
- Asegúrate de tener buena iluminación
- Mantén todo tu cuerpo visible en la cámara
- Verifica que la cámara funcione

### "Blender no recibe datos"
- Verifica que ambos scripts usen el mismo puerto (9000)
- Confirma que el receptor esté ejecutándose en Blender
- Checa el firewall de Windows

### "Firebase no funciona"
- Revisa `FIREBASE_SETUP.md`
- Ejecuta `python test_firebase_connection.py`
- El sistema funciona sin Firebase (solo guarda local)

---

## 📞 Recursos

- **MediaPipe Docs:** https://google.github.io/mediapipe/solutions/pose
- **Blender Python API:** https://docs.blender.org/api/current/
- **Firebase Admin SDK:** https://firebase.google.com/docs/admin/setup
- **Quaternions:** https://en.wikipedia.org/wiki/Quaternion

---

**Última actualización:** 22 de noviembre, 2025  
**Versión:** 2.0 (con Firebase completo)