import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag.textsplitter import chunk_transcript
from prompt.template import create_summary_prompt, create_qa_prompt_template
from chain.chain_builder import summary_chain, qa_chain
from rag.faiss import create_faiss_index, generate_answer
import re
from youtube_transcript_api import YouTubeTranscriptApi 
import re

def get_video_id(url):
    pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_transcript(url):
    video_id = get_video_id(url)
    
    ytt_api = YouTubeTranscriptApi()
    
    transcripts = ytt_api.list(video_id)
    
    transcript = ""
    for t in transcripts:
        if t.language_code == 'en':
            if t.is_generated:
                if len(transcript) == 0:
                    transcript = t.fetch()
            else:
                transcript = t.fetch()
                break 
 
    return transcript if transcript else None

def process(transcript):
    txt = ""

    for i in transcript:
        try:
            txt += f"Text: {i.text} Start: {i.start}\n"
        except KeyError:
            pass

    return txt

def summarize_video(video_url, llm):
    processed_transcript = ""
    if video_url:
        fetched_transcript = get_transcript(video_url)
        processed_transcript = process(fetched_transcript)
    else:
        return "Please provide a valid YouTube URL."
    if processed_transcript:
        summary_prompt = create_summary_prompt()
        summary_chain_instance = summary_chain(llm, summary_prompt)
        summary = summary_chain_instance.invoke({"transcript": processed_transcript})
        return summary
    else:
        return "No transcript available."


def answer_question(video_url, user_question, llm, em):
    processed_transcript = ""
    global fetched_transcript
    if not processed_transcript:
        if video_url:
            # Fetch and preprocess transcript
            fetched_transcript = get_transcript(video_url)
            processed_transcript = process(fetched_transcript)
        else:
            return "Please provide a valid YouTube URL."
        chunks = chunk_transcript(processed_transcript)
        faiss_index = create_faiss_index(chunks, em)
        qa_prompt = create_qa_prompt_template()
        qa_chain_instance = qa_chain(llm, qa_prompt)
        answer = generate_answer(user_question, faiss_index, qa_chain_instance)
        return answer
    else:
        return "Please provide a valid question and ensure the transcript is available."

if __name__ == "__main__":
    #url = input("Enter the YouTube video URL: ")
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    from llm.llm_client import get_llm
    llm = get_llm()
    print(summarize_video(url, llm))
