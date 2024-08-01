import streamlit as st
from langchain_openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import StringIO

# LLM y función de carga de llaves
def load_LLM(openai_api_key):
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    return llm

# Título y cabecera de la página
st.set_page_config(page_title="Resumidor de textos largos AI")
st.header("Resumidor de Textos Largos AI")

# Introducir la clave API de OpenAI
st.markdown("## Introduzca su clave API de OpenAI")

def get_openai_api_key():
    input_text = st.text_input(label="OpenAI API Key", placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input", type="password")
    return input_text

openai_api_key = get_openai_api_key()

# Entrada
st.markdown("## Cargue el archivo de texto que desea resumir")

uploaded_file = st.file_uploader("Elija un archivo", type="txt")

# Salida
st.markdown("### Este es su resumen:")

if uploaded_file is not None:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    file_input = stringio.read()

    if len(file_input.split(" ")) > 20000:
        st.write("Por favor, introduzca un archivo más corto. La longitud máxima es de 20000 palabras.")
        st.stop()

    if file_input:
        if not openai_api_key:
            st.warning('Introduzca la clave API de OpenAI.', icon="⚠️")
            st.stop()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], 
        chunk_size=5000, 
        chunk_overlap=350
    )

    splitted_documents = text_splitter.create_documents([file_input])

    llm = load_LLM(openai_api_key=openai_api_key)

    # Función para resumir en español
    def summarize_in_spanish(text):
        prompt = f"Por favor, resume el siguiente texto en español:\n\n{text}"
        response = llm(prompt)
        return response['choices'][0]['text'].strip()

    # Resumir cada fragmento por separado
    fragment_summaries = [summarize_in_spanish(doc) for doc in splitted_documents]

    # Unir todos los fragmentos en un solo resumen
    combined_summary_prompt = "Por favor, resume el siguiente texto en español:\n\n" + "\n\n".join(fragment_summaries)
    final_summary = summarize_in_spanish(combined_summary_prompt)

    st.write(final_summary)