from core.yt import answer_question
from llm.llm_client import get_llm, embedding_model

llm = get_llm()
embedding_model = embedding_model()

def main():
    answer_question("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "What is the main topic of the video?", llm=llm, em=embedding_model)
    print(answer_question)


if __name__ == "__main__":
    main()