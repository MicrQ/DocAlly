# assistant/views.py
import chromadb
from chromadb.utils import embedding_functions
from django.shortcuts import render, redirect, get_object_or_404
from .models import Document, ChatSession, Message
from .forms import UploadForm
from django.conf import settings
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
import google.generativeai as genai


def extract_text(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def process_document(doc_obj, gemini_api_key):
    # Extract text
    text = extract_text(doc_obj.file.path)

    # Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    # Create Chroma collection
    chroma_client = chromadb.Client()
    embedding_func = embedding_functions.GoogleGenerativeAIEmbeddingFunction(
        api_key=gemini_api_key,
        model_name="models/embedding-001"
    )
    collection = chroma_client.create_collection(
        name=f"doc_{doc_obj.id}",
        embedding_function=embedding_func
    )

    # Add chunks
    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[f"{doc_obj.id}_{i}"],
            documents=[chunk],
            metadatas=[{"chunk": i}]
        )

    doc_obj.processed = True
    doc_obj.save()
    return collection


def upload_view(request):
    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            doc = form.save(commit=False)
            doc.filename = doc.file.name
            doc.save()
            gemini_api_key = form.cleaned_data['gemini_api_key']

            # Process doc & start chat session
            process_document(doc, gemini_api_key)
            session = ChatSession.objects.create(document=doc, gemini_api_key=gemini_api_key)
            return redirect("chat", session_id=session.id)
    else:
        form = UploadForm()
    return render(request, "assistant/upload.html", {"form": form})


def chat_view(request, session_id):
    session = get_object_or_404(ChatSession, id=session_id)
    chroma_client = chromadb.Client()
    collection = chroma_client.get_collection(name=f"doc_{session.document.id}")
    messages = session.messages.all().order_by("created_at")

    if request.method == "POST":
        user_msg = request.POST.get("message")
        Message.objects.create(session=session, role="user", text=user_msg)

        # Retrieve relevant chunks
        results = collection.query(query_texts=[user_msg], n_results=3)
        context_chunks = "\n".join(results["documents"][0])

        # Call Gemini
        genai.configure(api_key=session.gemini_api_key)
        model = genai.GenerativeModel("gemini-pro")
        prompt = f"Answer the question based on the following document context:\n\n{context_chunks}\n\nQuestion: {user_msg}"
        response = model.generate_content(prompt)
        answer = response.text

        Message.objects.create(session=session, role="assistant", text=answer)
        return redirect("chat", session_id=session.id)

    return render(request, "assistant/chat.html", {"session": session, "messages": messages})
