from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter


path = ("/doc/*")
loader = PyPDFLoader(path)
pages = loader.load_and_split()

# Objeto que va a hacer los cortes en el texto
split = CharacterTextSplitter(chunk_size=300, separator = '.\n')