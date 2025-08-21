import os
import mimetypes
import time
import tempfile
from google.colab import userdata, files  

# --- Core Logic & AI Imports ---
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import ArxivQueryRun, DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# --- Output & Utility Imports ---
from fpdf import FPDF
from gtts import gTTS
from PIL import Image
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager


try:
    os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
except userdata.SecretNotFoundError:
    raise ValueError("SECRET NOT FOUND: Please add 'GOOGLE_API_KEY' to your Colab Secrets.")


# ======================================================
# Agent and Processing Functions
# ======================================================


def select_tool_and_retrieve(llm, prompt: str) -> list[str]:
    print("Step 1: Selecting tool and retrieving documents...")
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
    web_search = DuckDuckGoSearchRun()

    # Check if sample.pdf exists before creating the tool
    pdf_tool_description = "This tool is currently disabled because sample.pdf was not found."
    if os.path.exists("sample.pdf"):
        pdf_tool_description = "Use this to read a known PDF document named sample.pdf."

    pdf_tool = Tool(
        name="PDF Document Reader",
        func=lambda q: "\n".join([doc.page_content for doc in PyPDFLoader("sample.pdf").load()]) if os.path.exists("sample.pdf") else "File not found.",
        description=pdf_tool_description
    )
    tools = [wikipedia, arxiv, web_search, pdf_tool]
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True)
    try:
        result = agent.invoke({"input": prompt})
        return [result['output']]
    except Exception as e:
        print(f"An error occurred during tool retrieval: {e}")
        return []

def chunk_documents(documents: list[str]) -> list[str]:
    print("\nStep 2: Chunking documents...")
    if not documents or not documents[0]:
        print("Warning: No documents to chunk.")
        return []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_text(documents[0])
    print(f"Created {len(chunks)} chunks.")
    return chunks

def embed_and_search(chunks: list[str], query: str) -> list[Document]:
    print("\nStep 3: Embedding chunks and performing similarity search...")
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    document_objects = [Document(page_content=chunk) for chunk in chunks]
    vector_db = Chroma.from_documents(document_objects, embedding_model)
    results = vector_db.similarity_search(query, k=4)
    print(f"Found {len(results)} relevant chunks for summarization.")
    return results

def generate_summary_and_keywords(llm, query: str, top_chunks: list[Document]) -> dict:
    print("\nStep 4: Generating summary and keywords...")
    context = "\n\n---\n\n".join([doc.page_content for doc in top_chunks])
    template = "..." # The prompt template is the same, keeping it short here for brevity
    template = """
    Based on the context below, answer the user's original query.
    Original Query: {query}
    Context:
    {context}
    Your Task:
    1.  Write a clear summary (3-5 sentences) that directly addresses the query.
    2.  Suggest one or two strong keywords (a noun or short phrase) that represent the main topic.
    Format your response exactly as follows:
    Summary: [Your summary here]
    Keywords: [Your keyword(s) here]
    """
    prompt_template = PromptTemplate(input_variables=["query", "context"], template=template)
    chain = LLMChain(llm=llm, prompt=prompt_template)
    try:
        response = chain.invoke({"query": query, "context": context})['text']
        summary_part = response.split("Summary:")[1].split("Keywords:")[0].strip()
        keywords_part = response.split("Keywords:")[1].strip()
        return {"summary": summary_part, "keywords": keywords_part}
    except Exception:
        return {"summary": "Could not generate a summary.", "keywords": "None"}

# ======================================================
# Content Generation and Output Functions for Colab
# ======================================================

def get_images_colab(keywords: str, download_path="images", limit=2) -> list[str]:
    print(f"\nStep 5: Scraping images for keywords: '{keywords}'...")
    if not os.path.exists(download_path): os.makedirs(download_path)

    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    # Use webdriver_manager to handle the Chrome driver in Colab
    try:
        driver = webdriver.Chrome(options=options)
        search_query = f"https://www.google.com/search?q={keywords.replace(' ', '+')}&tbm=isch"
        driver.get(search_query)
        time.sleep(3) # Give more time for Colab's network
        images = driver.find_elements(By.CSS_SELECTOR, "img.rg_i")[:limit]
        saved_files = []
        for i, img in enumerate(images):
            try:
                src = img.get_attribute("src")
                if not src or not src.startswith('http'): continue
                response = requests.get(src, stream=True, timeout=5)
                response.raise_for_status()
                ext = mimetypes.guess_extension(response.headers.get("content-type", "").split(";")[0]) or ".jpg"
                file_path = os.path.join(download_path, f"{keywords.replace(' ', '_')}_{i+1}{ext}")
                with open(file_path, "wb") as f: f.write(response.content)
                saved_files.append(file_path)
            except Exception as e: print(f"Could not download image {i+1}: {e}")
        print(f"Successfully saved {len(saved_files)} images.")
        return saved_files
    finally:
        if 'driver' in locals(): driver.quit()

def create_pdf(summary_text: str, images: list[str], output_filename="output.pdf") -> str:
    print(f"\nStep 6: Creating PDF document '{output_filename}'...")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(0, 10, "Research Summary Report", ln=True, align='C')
    pdf.ln(10)
    page_width = pdf.w - 2 * pdf.l_margin
    if images and os.path.exists(images[0]):
        try:
            pdf.image(images[0], x=pdf.l_margin, w=page_width)
            pdf.ln(10)
        except Exception as e: print(f"Could not add image {images[0]} to PDF: {e}")
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, summary_text)
    pdf.output(output_filename)
    print(f"PDF saved successfully.")
    return output_filename

def text_to_speech_colab(text: str) -> str:
    """
    COLAB-FIX: Generates speech and saves it to a file.
    It returns the filename so the user can download it. `playsound` is removed.
    """
    print("\nStep 7: Generating audio speech file...")
    try:
        tts = gTTS(text=text, lang="en")
        filename = "summary_audio.mp3"
        tts.save(filename)
        print(f"Audio file '{filename}' created. You can download it from the file browser.")
        return filename
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        return None

# ======================================================
# Step 4: Orchestration and Execution
# ======================================================

def run_research_agent_colab():



  agent_llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.2)
  summary_llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.7)

    

  prompt = input("Please enter the topic you want to research: ")
  if not prompt:
      print("No input received. Exiting.")
      return

  documents = select_tool_and_retrieve(agent_llm, prompt)
  if not documents: return

  chunks = chunk_documents(documents)
  if not chunks: return

  top_chunks = embed_and_search(chunks, prompt)
  if not top_chunks: return

  result = generate_summary_and_keywords(summary_llm, prompt, top_chunks)
  summary = result.get('summary', 'Failed to generate summary.')
  keywords = result.get('keywords', 'general')

  print("\n--- Final Generated Content ---")
  print(f"SUMMARY:\n{summary}")
  print(f"\nKEYWORDS:\n{keywords}")
  print("--------------------------------")

  if "Could not generate" in summary:
      return

  images = get_images_colab(keywords)
  pdf_file = create_pdf(summary, images)
  print(f"\n✅ PDF report '{pdf_file}' has been created in the file browser.")

  listen_answer = input("Would you like me to generate an audio file of the summary? (yes/no): ").lower()
  if listen_answer == "yes":
      audio_file = text_to_speech_colab(summary)
      if audio_file:
          print(f"✅ Audio file '{audio_file}' has been created.")
          files.download(audio_file) 

  print("\nResearch complete.")

# --- Start the agent ---
run_research_agent_colab()