import streamlit as st
from langchain import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from io import StringIO

#LLM y función de carga de llaves
def load_LLM(openai_api_key):
    # Asegúrese de que su openai_api_key se establece como una variable de entorno
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    return llm


#Título y cabecera de la página
st.set_page_config(page_title="Resumidor de textos largos AI")
st.header("Resumidor de Textos Largos AI")


#Intro: instrucciones
col1, col2 = st.columns(2)

with col1:
    st.markdown("ChatGPT no puede resumir textos largos. Ahora puedes hacerlo con esta aplicación.")

with col2:
    st.write("Contacte con [Matias Toro Labra](https://www.linkedin.com/in/luis-matias-toro-labra-b4074121b/) para construir sus proyectos de IA")


#Introducir la clave API de OpenAI
st.markdown("## Introduzca su clave API de OpenAI")

def get_openai_api_key():
    input_text = st.text_input(label="OpenAI API Key ",  placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input", type="password")
    return input_text

openai_api_key = get_openai_api_key()


# Entrada
st.markdown("## Cargue el archivo de texto que desea resumir")

uploaded_file = st.file_uploader("Elija un archivo", type="txt")

       
# Salida
st.markdown("### Este es su resumen:")

if uploaded_file is not None:
    # Leer el archivo como bytes
    bytes_data = uploaded_file.getvalue()
    
    # Convertir a una cadena basada en IO
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    
    # Leer el archivo como cadena
    string_data = stringio.read()
    
    file_input = string_data

    if len(file_input.split(" ")) > 20000:
        st.write("Por favor, introduce un archivo más corto. La longitud máxima es de 20000 palabras.")
        st.stop()

    if file_input:
        if not openai_api_key:
            st.warning('Introduce la clave API de OpenAI. \
            Instrucciones [aquí](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', 
            icon="⚠️")
            st.stop()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], 
        chunk_size=5000, 
        chunk_overlap=350
    )

    splitted_documents = text_splitter.create_documents([file_input])

    # Inicializar el modelo de lenguaje con la clave API
    llm = OpenAI(api_key=openai_api_key)

    # Cargar la cadena de resumen
    summarize_chain = load_summarize_chain(
        llm=llm, 
        chain_type="map_reduce"
    )

    # Preparar los documentos para el resumen
    documents = [{"page_content": doc.page_content} for doc in splitted_documents]

    try:
        summary_output = summarize_chain.run(documents)
        st.write(summary_output)
    except Exception as e:
        st.error(f"Error al ejecutar la cadena de resumen: {str(e)}")