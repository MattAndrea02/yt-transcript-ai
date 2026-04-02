from operator import itemgetter
import os
import sys
from langchain_core.output_parsers import StrOutputParser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from langchain_core.runnables import RunnablePassthrough
from prompt.template import create_summary_prompt, create_qa_prompt_template

def summary_chain(llm, output_parser):
    prompt = create_summary_prompt() 
    return {"transcript": itemgetter("transcript")} | prompt | llm | StrOutputParser()

def qa_chain(llm, output_parser):
    prompt = create_qa_prompt_template()
    return {"context": itemgetter("context"), "question": itemgetter("question")} | prompt | llm | output_parser
