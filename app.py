import urllib.request 
import fitz
import re
import numpy as np
import tensorflow_hub as hub
import openai
import gradio as gr
import os
from sklearn.neighbors import NearestNeighbors

def download_pdf(url, output_path):
    urllib.request.urlretrieve(url, output_path)


def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text


def pdf_to_text(path, start_page=1, end_page=None):
    doc = fitz.open(path)
    total_pages = doc.page_count

    if end_page is None:
        end_page = total_pages

    text_list = []

    for i in range(start_page-1, end_page):
        text = doc.load_page(i).get_text("text")
        text = preprocess(text)
        text_list.append(text)

    doc.close()
    return text_list


def text_to_chunks(texts, word_length=150, start_page=1):
    text_toks = [t.split(' ') for t in texts]
    page_nums = []
    chunks = []
    
    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i:i+word_length]
            if (i+word_length) > len(words) and (len(chunk) < word_length) and (
                len(text_toks) != (idx+1)):
                text_toks[idx+1] = chunk + text_toks[idx+1]
                continue
            chunk = ' '.join(chunk).strip()
            chunk = f'[Page no. {idx+start_page}]' + ' ' + '"' + chunk + '"'
            chunks.append(chunk)
    return chunks


class SemanticSearch:
    
    def __init__(self):
        self.use = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
        self.fitted = False
    
    
    def fit(self, data, batch=1000, n_neighbors=5):
        self.data = data
        self.embeddings = self.get_text_embedding(data, batch=batch)
        n_neighbors = min(n_neighbors, len(self.embeddings))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn.fit(self.embeddings)
        self.fitted = True
    
    
    def __call__(self, text, return_data=True):
        inp_emb = self.use([text])
        neighbors = self.nn.kneighbors(inp_emb, return_distance=False)[0]
        
        if return_data:
            return [self.data[i] for i in neighbors]
        else:
            return neighbors
    
    
    def get_text_embedding(self, texts, batch=1000):
        embeddings = []
        for i in range(0, len(texts), batch):
            text_batch = texts[i:(i+batch)]
            emb_batch = self.use(text_batch)
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)
        return embeddings



def load_recommender(path, start_page=1):
    global recommender
    texts = pdf_to_text(path, start_page=start_page)
    chunks = text_to_chunks(texts, start_page=start_page)
    recommender.fit(chunks)
    return 'Corpus Loaded.'

def generate_text(openAI_key, prompt, model="gpt-3.5-turbo"):
    openai.api_key = openAI_key
    temperature=0.7
    max_tokens=256
    top_p=1
    frequency_penalty=0
    presence_penalty=0

    if model == "text-davinci-003":
        completions = openai.Completion.create(
            engine=model,
            prompt=prompt,
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=temperature,
        )
        message = completions.choices[0].text
    else:
        message = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "assistant", "content": "Here is some initial assistant message."},
                {"role": "user", "content": prompt}
            ],
            temperature=.3,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        ).choices[0].message['content']
    return message

  
def generate_answer(question, openAI_key, model):
    topn_chunks = recommender(question)
    prompt = 'search results:\n\n'
    for c in topn_chunks:
        prompt += c + '\n\n'
        
    prompt += "Instructions: Compose a comprehensive reply to the query using the search results given. "\
              "Cite each reference using [ Page Number] notation. "\
              "Only answer what is asked. The answer should be short and concise. \n\nQuery: "
    
    prompt += f"{question}\nAnswer:"
    answer = generate_text(openAI_key, prompt, model)
    return answer


def question_answer(chat_history, url, file, question, openAI_key, model):
    try:
        if openAI_key.strip()=='':
            return '[ERROR]: Please enter your Open AI Key. Get your key here : https://platform.openai.com/account/api-keys'
        if url.strip() == '' and file is None:
            return '[ERROR]: Both URL and PDF is empty. Provide at least one.'
        if url.strip() != '' and file is not None:
            return '[ERROR]: Both URL and PDF is provided. Please provide only one (either URL or PDF).'
        if model is None or model =='':
            return '[ERROR]: You have not selected any model. Please choose an LLM model.'
        if url.strip() != '':
            glob_url = url
            download_pdf(glob_url, 'corpus.pdf')
            load_recommender('corpus.pdf')
        else:
            old_file_name = file.name
            file_name = file.name
            file_name = file_name[:-12] + file_name[-4:]
            os.rename(old_file_name, file_name)
            load_recommender(file_name)
        if question.strip() == '':
            return '[ERROR]: Question field is empty'
        if model == "text-davinci-003" or model == "gpt-4" or model == "gpt-4-32k":
            answer = generate_answer_text_davinci_003(question, openAI_key)
        else:
            answer = generate_answer(question, openAI_key, model)
        chat_history.append([question, answer])
        return chat_history
    except openai.error.InvalidRequestError as e:
        return f'[ERROR]: Either you do not have access to GPT4 or you have exhausted your quota!'



def generate_text_text_davinci_003(openAI_key,prompt, engine="text-davinci-003"):
    openai.api_key = openAI_key
    completions = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=512,
        n=1,
        stop=None,
        temperature=0.7,
    )
    message = completions.choices[0].text
    return message


def generate_answer_text_davinci_003(question,openAI_key):
    topn_chunks = recommender(question)
    prompt = ""
    prompt += 'search results:\n\n'
    for c in topn_chunks:
        prompt += c + '\n\n'
        
    prompt += "Instructions: Compose a comprehensive reply to the query using the search results given. "\
              "Cite each reference using [ Page Number] notation (every result has this number at the beginning). "\
              "Citation should be done at the end of each sentence. If the search results mention multiple subjects "\
              "with the same name, create separate answers for each. Only include information found in the results and "\
              "don't add any additional information. Make sure the answer is correct and don't output false content. "\
              "If the text does not relate to the query, simply state 'Found Nothing'. Ignore outlier "\
              "search results which has nothing to do with the question. Only answer what is asked. The "\
              "answer should be short and concise. \n\nQuery: {question}\nAnswer: "
    
    prompt += f"Query: {question}\nAnswer:"
    answer = generate_text_text_davinci_003(openAI_key, prompt,"text-davinci-003")
    return answer

# pre-defined questions
questions = [
    "What did the study investigate?",
    "Can you provide a summary of this paper?",
    "what are the methodologies used in this study?",
    "what are the data intervals used in this study? Give me the start dates and end dates?",
    "what are the main limitations of this study?",
    "what are the main shortcomings of this study?",
    "what are the main findings of the study?",
    "what are the main results of the study?",
    "what are the main contributions of this study?",
    "what is the conclusion of this paper?",
    "what are the input features used in this study?",
    "what is the dependent variable in this study?",
]


recommender = SemanticSearch()

title = 'PDF GPT Turbo'
description = """ PDF GPT Turbo allows you to chat with your PDF files. It uses Google's Universal Sentence Encoder with Deep averaging network (DAN) to give hallucination free response by improving the embedding quality of OpenAI. It cites the page number in square brackets([Page No.]) and shows where the information is located, adding credibility to the responses."""

with gr.Blocks(css="""#chatbot { font-size: 14px; min-height: 1200; }""") as demo:

    gr.Markdown(f'<center><h3>{title}</h3></center>')
    gr.Markdown(description)

    with gr.Row():
        
        with gr.Group():
            gr.Markdown(f'<p style="text-align:center">Get your Open AI API key <a href="https://platform.openai.com/account/api-keys">here</a></p>')
            with gr.Accordion("API Key"):
                openAI_key = gr.Textbox(label='Enter your OpenAI API key here', password=True)
                url = gr.Textbox(label='Enter PDF URL here   (Example: https://arxiv.org/pdf/1706.03762.pdf )')
                gr.Markdown("<center><h4>OR<h4></center>")
                file = gr.File(label='Upload your PDF/ Research Paper / Book here', file_types=['.pdf'])
            question = gr.Textbox(label='Enter your question here')
            gr.Examples(
                [[q] for q in questions],
                inputs=[question],
                label="PRE-DEFINED QUESTIONS: Click on a question to auto-fill the input box, then press Enter!",
            )
            model = gr.Radio([
                'gpt-3.5-turbo', 
                'gpt-3.5-turbo-16k', 
                'gpt-3.5-turbo-0613', 
                'gpt-3.5-turbo-16k-0613', 
                'text-davinci-003',
                'gpt-4',
                'gpt-4-32k'
            ], label='Select Model', default='gpt-3.5-turbo')
            btn = gr.Button(value='Submit')

            btn.style(full_width=True)

        with gr.Group():
            chatbot = gr.Chatbot(placeholder="Chat History", label="Chat History", lines=50, elem_id="chatbot")


#
    # Bind the click event of the button to the question_answer function
    btn.click(
        question_answer,
        inputs=[chatbot, url, file, question, openAI_key, model],
        outputs=[chatbot],
    )

demo.launch()


