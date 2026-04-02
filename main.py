from core.yt import answer_question
from llm.llm_client import get_llm, embedding_model

llm = get_llm()
embedding_model = embedding_model()

def main():
    url = input("Insert URL: ")
    question = input("Question: ")
    answer = answer_question(url, question, llm=llm, em=embedding_model)
    print(answer)


if __name__ == "__main__":
    main()