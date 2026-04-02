from langchain_core.prompts import PromptTemplate

def create_summary_prompt():
    template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an AI assistant tasked with summarizing YouTube video transcri
    Instructions:
    1. Summarize the transcript in a single concise paragraph.
    2. Ignore any timestamps in your summary.
    3. Focus on the spoken content (Text) of the video.
    Note: In the transcript, "Text" refers to the spoken words in the vide
    Please summarize the following YouTube video transcript:
    {transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    # Create the PromptTemplate object with the defined template
    prompt = PromptTemplate(
    input_variables=["transcript"],
    template=template
    )

    return prompt

def create_qa_prompt_template():
    """
    Create a PromptTemplate for question answering based on video content.
    Returns:
    PromptTemplate: A PromptTemplate object configured for Q&A tasks.
    """

    # Define the template string
    qa_template = """
    You are an expert assistant providing detailed answers based on the fo
    Relevant Video Context: {context}
    Based on the above context, please answer the following question:
    Question: {question}
    """
    # Create the PromptTemplate object
    prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=qa_template
    )
    return prompt_template