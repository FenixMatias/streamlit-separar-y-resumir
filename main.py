import streamlit as st
from io import StringIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import load_summarize_chain
from langchain_openai import OpenAI

# Asegúrate de tener tu clave API de OpenAI
openai_api_key = st.text_input("Introduce tu clave API de OpenAI", type="password")

uploaded_file = st.file_uploader("Sube tu archivo")

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
        chain_type="map_reduce",
        llm_kwargs={"language": "es"}  # Especificar el idioma español en los parámetros del modelo
    )

    # Preparar el prompt en español
    prompt = "Por favor, resume el siguiente texto en español:\n\n" + file_input

    summary_output = summarize_chain.run([prompt])

    st.write(summary_output)