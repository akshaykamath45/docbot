import streamlit as st
import openai
from core import get_index_for_pdf
import os
from dotenv import load_dotenv


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


st.title("DocBot")

@st.cache_resource
def create_vectordb(files, filenames):
    with st.spinner("Vector database"):
        vectordb = get_index_for_pdf(
            [file.getvalue() for file in files], filenames, openai.api_key
        )
    return vectordb

pdf_files = st.file_uploader("", type="pdf", accept_multiple_files=True)

if pdf_files:
    pdf_file_names = [file.name for file in pdf_files]
    st.session_state["vectordb"] = create_vectordb(pdf_files, pdf_file_names)

prompt = st.session_state.get("prompt", [{"role": "system", "content": "none"}])

for message in prompt:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

prompt_template = """
    You are a helpful Assistant who answers to users questions based on multiple contexts given to you.
    Keep your answer short and to the point.
    The evidence are the context of the pdf extract with metadata. 
    Carefully focus on the metadata specially 'filename' and 'page' whenever answering.
    Make sure to add filename and page number at the end of sentence you are citing to.
    Reply "Not applicable" if text is irrelevant.
    The PDF content is:
    {pdf_extract}
"""

def get_response(prompt):
    try:
        stream = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=prompt,
            temperature=1,
            max_tokens=1000,
            stop=None,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                response_text = chunk.choices[0].delta.content
                yield response_text
        if response_text:
            return response_text
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"status": False, "error": "Something went wrong"}


# handling user's question
question = st.chat_input("Ask anything")
if question:
    vectordb = st.session_state.get("vectordb", None)
    if not vectordb:
        with st.message("assistant"):
            st.write("You need to provide a PDF")
            st.stop()

    # searching the vectordb for similar content
    search_results = vectordb.similarity_search(question, k=3)
    pdf_extract = "\n ".join([result.page_content for result in search_results])

    prompt[0] = {
        "role": "system",
        "content": prompt_template.format(pdf_extract=pdf_extract),
    }
   
    prompt.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        botmsg = st.empty()
        st.session_state['botmsg'] = botmsg

    result = get_response(prompt)
    st.session_state['result'] = result
    st.session_state['botmsg'].write(result)