# app.py

import streamlit as st
import tempfile
import os
from main import process_audio, process_pdf, extract_frames_from_video, process_video_frames, create_vector_db, query_vector_database

st.set_page_config(page_title="Multimodal QA System", layout="wide")
st.title("📄🧠 Multimodal Document and Media QA System")

file_type = st.selectbox("Choose file type", ["PDF", "Audio", "Video"])

uploaded_file = st.file_uploader("Upload a file", type=["pdf", "mp3", "mp4"])

query = st.text_input("Enter your query")

if st.button("Submit Query"):
    if uploaded_file is not None and query:
        original_filename = uploaded_file.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        try:
            if file_type == "Audio" and uploaded_file.name.endswith(".mp3"):
                st.info("Processing audio file...")
                documents = process_audio(tmp_file_path)
                for doc in documents:
                    doc.metadata["audio_file_name"] = original_filename

            elif file_type == "PDF" and uploaded_file.name.endswith(".pdf"):
                st.info("Processing PDF file...")
                documents = process_pdf(tmp_file_path)
                for doc in documents:
                    doc.metadata["pdf_file_name"] = original_filename

            elif file_type == "Video" and uploaded_file.name.endswith(".mp4"):
                st.info("Extracting video frames...")
                frame_dir = os.path.join(tempfile.gettempdir(), "video_frames")
                extract_frames_from_video(tmp_file_path, frame_dir)
                st.info("Processing video frames...")
                documents = process_video_frames(frame_dir)
                for doc in documents:
                    doc.metadata["video_name"] = os.path.splitext(original_filename)[0]

            else:
                st.error("File extension does not match selected file type.")
                st.stop()

            db = create_vector_db(documents)
            st.success("Vector database created!")

            st.info("Querying the vector database...")
            result = query_vector_database(db, query)

            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Related Metadata")
            for meta in result["metadata"]:
                if meta["file_name"]:
                    st.write(f"**File Name:** {meta['file_name']}")
                if meta["file_path"]:
                    st.write(f"**File Path:** {meta['file_path']}")
                if meta["video_name"]:
                    st.write(f"**Video Name:** {meta['video_name']}")
                st.markdown("---")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload a file and enter a query.")


# streamlit run app.py