from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
import tensorflow_hub as hub
import pandas as pd
import tensorflow as tf
import os

path = ("/doc/*")
loader = PyPDFLoader(path)
pages = loader.load_and_split()

# Objeto que va a hacer los cortes en el texto
split = CharacterTextSplitter(chunk_size=300, separator = '.\n')

textos = split.split_documents(pages) # Lista de textos

# Extraemos la parte de page_content de cada texto y lo pasamos a un dataframe
textos = [str(i.page_content) for i in textos] #Lista de parrafos
parrafos = pd.DataFrame(textos, columns=["texto"])


# Cargar el modelo Universal Sentence Encoder (versión 4)
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Obtener embeddings para los párrafos
try:
    embeddings = embed(parrafos["texto"])
except Exception as e:
    print(f"Error al obtener embeddings: {e}")
    exit()

# Agregar los embeddings al DataFrame
parrafos['Embedding'] = tf.convert_to_tensor(embeddings).numpy().tolist()

# Guardar el DataFrame con los embeddings en un archivo CSV (verifica si existe)
nombre_archivo = "emb-SI.csv"
if os.path.exists(nombre_archivo):
    nombre_archivo = f"{nombre_archivo}-{int(time.time())}"

parrafos.to_csv(nombre_archivo, index=False)