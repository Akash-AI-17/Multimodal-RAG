
import os
import whisper
import base64
import cv2
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from groq import Groq
import torch
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import whisper
import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.vectorstores import FAISS
from groq import Groq
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Initialize Whisper model
whisper_model = whisper.load_model("base")

llm = ChatGroq(
   model_name="llama-3.1-8b-instant",
   api_key="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",
   temperature = 0.4
)

# Function to process audio files
def process_audio(file_path):
    if file_path.endswith(".mp3"):  # Process only .mp3 files
        # Transcribe the audio file
        transcription = whisper_model.transcribe(file_path, fp16=False)["text"].strip()

        # Extract the file name from the path
        file_name = os.path.basename(file_path)

        # Create a Document with transcription and audio reference
        document = Document(
            page_content=transcription,
            metadata={"source_type": "audio", "audio_file_name": file_name, "audio_file_path": file_path}
        )
        return [document]

# Function to process PDF files
def process_pdf(pdf_file):
    loader = PyPDFLoader(pdf_file)
    pdf_docs = loader.load()

    # Add metadata for each PDF document
    for doc in pdf_docs:
        doc.metadata.update({"source_type": "pdf", "pdf_file_name": os.path.basename(pdf_file), "pdf_file_path": pdf_file})
    return pdf_docs

# Function to extract frames from video and process them
def extract_frames_from_video(video_path, output_folder):
    # Clear the output folder if it already exists
    if os.path.exists(output_folder):
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps == 0:
        print("Unable to retrieve FPS from video. Exiting.")
        return []

    duration = frame_count / fps
    success, image = video_capture.read()
    frame_number = 0
    frames = []

    while success:
        # Save the first frame, every 5 seconds, and the last frame
        if frame_number == 0 or frame_number % int(fps * 5) == 0 or frame_number == frame_count - 1:
            frame_time = frame_number / fps
            output_frame_filename = os.path.join(output_folder, f'frame_{int(frame_time)}.jpg')
            cv2.imwrite(output_frame_filename, image)
            frames.append(output_frame_filename)

        success, image = video_capture.read()
        frame_number += 1

    video_capture.release()
    return frames

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function to process video frames and create vector database
def process_video_frames(image_directory):
    if not os.path.exists(image_directory):
        raise FileNotFoundError(f"The directory {image_directory} does not exist.")

    # Extract video name from directory path
    video_name = os.path.basename(image_directory)

    # Initialize Groq client
    client = Groq(api_key="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")  # Replace with your API key

    # List to store Document instances
    documents = []

    # Iterate through all images in the directory
    for filename in os.listdir(image_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(image_directory, filename)

            # Get the base64 string
            base64_image = encode_image(image_path)

            # Create chat completion for the current image
            chat_completion = client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"What's in this image? ({filename})"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }],
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
            )

            # Create a Document from the output and store it in the documents list
            doc = Document(
                page_content=chat_completion.choices[0].message.content,
                metadata={
                    "image_filename": filename,
                    "video_name": video_name,
                }
            )
            documents.append(doc)

    return documents

# Function to split documents into chunks and create a vector database
def create_vector_db(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(documents)

    # Create embeddings and vector database
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    db = FAISS.from_documents(texts, embeddings)
    db.save_local("vector_db")

    return db

def query_vector_database(db, question):
    results = db.similarity_search(question, k=5)
    source_knowledge = "\n".join([doc.page_content for doc in results])
    metadata = [doc.metadata for doc in results]

    prompt = f"""
    [INST]<<SYS>>
    Based on this example, complete the task below.<</SYS>>
    Context: {source_knowledge}
    Question: {question}
    Metadata: {metadata}

    Answer based on the context:
    [/INST]
    """

    # Assuming `llm.invoke` to generate the final response
    llm_response = llm.invoke(prompt)
    if hasattr(llm_response, "content"):
        answer = llm_response.content.strip()
    else:
        answer = str(llm_response).strip()

    # Prepare the final result
    res = {
        "answer": answer,
        "metadata": [
            {
                "file_name": doc.metadata.get("audio_file_name") or doc.metadata.get("pdf_file_name") or doc.metadata.get("image_filename", ""),
                "file_path": doc.metadata.get("audio_file_path") or doc.metadata.get("pdf_file_path") or "",
                "video_name": doc.metadata.get("video_name", ""),
            }
            for doc in results
        ],
    }

    # Remove duplicate metadata entries
    unique_metadata = []
    seen = set()
    for entry in res["metadata"]:
        metadata_tuple = (entry["file_name"], entry["file_path"], entry["video_name"])
        if metadata_tuple not in seen:
            unique_metadata.append(entry)
            seen.add(metadata_tuple)

    res["metadata"] = unique_metadata

    return res

