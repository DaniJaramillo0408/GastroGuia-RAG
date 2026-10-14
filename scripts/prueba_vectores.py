print("1. Arrancando script...")
import os
from datetime import datetime
print("2. Cargando librerias de base de datos...")
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId
print("3. Cargando motor de IA (Esto puede tardar un momento)...")
from sentence_transformers import SentenceTransformer
from PIL import Image
print("4. ¡Todas las librerias cargadas!")

# 1. Cargar variables de entorno (tu contraseña segura)
# ... de aquí en adelante dejas el resto de tu código intacto ...

# 1. Cargar variables de entorno (tu contraseña segura)
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")

# 2. Conectar a MongoDB Atlas
print("Conectando a la base de datos...")
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# 3. Cargar Modelos de IA
# La primera vez que ejecutes esto, descargará los modelos de internet (puede tardar un par de minutos).
print("Cargando modelo de texto (all-MiniLM-L6-v2)...")
text_model = SentenceTransformer('all-MiniLM-L6-v2') 

print("Cargando modelo de imágenes (CLIP)...")
image_model = SentenceTransformer('clip-ViT-B-32') 

# 4. Preparar datos de prueba
texto_ejemplo = "La hamburguesa clasica contiene carne de res, pan brioche, queso cheddar, lechuga y tomate."

# Generar embedding de texto real (384 dimensiones)
print("Generando vector de texto...")
vector_texto = text_model.encode(texto_ejemplo).tolist()

# Generar embedding de imagen real (512 dimensiones)
# Como aún no tienes fotos, crearemos una imagen roja temporal en la memoria solo para probar la IA
print("Generando vector de imagen...")
imagen_temporal = Image.new('RGB', (224, 224), color = 'red') 
vector_imagen = image_model.encode(imagen_temporal).tolist()

# 5. Insertar en Base de Datos cumpliendo tus estrictas reglas de validación
doc_id = ObjectId()
img_id = ObjectId()

# A. Insertar Documento
print("Guardando Documento...")
db.documentos.insert_one({
    "_id": doc_id,
    "titulo": "Hamburguesa clasica: prueba con IA",
    "idioma": "es",
    "fecha": datetime.utcnow(),
    "fuente": "Contenido elaborado y revisado por el grupo",
    "contenido_texto": texto_ejemplo,
    "metadatos": {
        "autor": "Equipo GastroGuia",
        "categoria": "productos",
        "etiquetas": ["hamburguesa", "prueba"]
    },
    "imagenes": [img_id]
})

# B. Insertar Chunk (con el vector de 384 valores)
print("Guardando Chunk...")
db.chunks.insert_one({
    "doc_id": doc_id,
    "chunk_index": 0,
    "estrategia_chunking": "fixed-size",
    "chunk_texto": texto_ejemplo,
    "embedding": vector_texto,
    "modelo": "all-MiniLM-L6-v2"
})

# C. Insertar Imagen (con el vector de 512 valores)
print("Guardando Imagen...")
db.imagenes.insert_one({
    "_id": img_id,
    "doc_id": doc_id,
    "ruta_archivo": "imagenes/temporal.jpg",
    "descripcion": "Imagen generada para prueba de CLIP",
    "etiquetas": ["prueba", "temporal"],
    "embedding": vector_imagen,
    "modelo": "clip-vit-base-patch32"
})

print("¡Éxito total! Revisa tu base de datos en MongoDB Compass, los vectores reales han sido guardados.")