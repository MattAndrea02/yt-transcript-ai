from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_transcript(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks