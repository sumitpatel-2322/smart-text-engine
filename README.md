# 🎬 Movie4U — Iteration 1  
**ML-powered movie review summarizer & sentiment analyzer built with FastAPI and Docker**

---

## 🚀 Project Motivation

This project started as an attempt to move beyond Jupyter notebooks and understand how **deep learning and transformer-based models behave in real applications**.

Instead of jumping directly into a recommendation system, **Iteration 1 focuses on building a clean, production-ready inference pipeline** for analyzing movie reviews — including summarization, sentiment prediction, and deployment considerations.

The goal was not just to “make a model work”, but to understand:
- how models should be structured outside notebooks  
- how inference pipelines should be organized  
- how ML systems behave inside Docker containers  

---

## 🧩 What Iteration 1 Does

This web application:

- Accepts raw **movie review text** from the user  
- Extracts **key sentences** using **extractive summarization**
- Predicts **sentiment** (`Positive / Negative`)
- Displays results through a simple web UI
- Runs fully inside a **Docker container**

❌ **This version does NOT recommend movies yet.**  
That will be part of future iterations.

---

## 🏗️ Architecture Overview

**Backend**
- FastAPI for API and application routing
- Modular inference layer (`classify`, `summarize`)
- Config-driven lifecycle management

**Frontend**
- HTML, CSS, and vanilla JavaScript
- Lightweight UI focused on functionality

**ML Inference**
- Transformer-based embeddings (DistilBERT)
- Custom sentiment classifier
- Extractive summarization using sentence embeddings
- Lazy or preloaded model loading (configurable)

**Containerization**
- Dockerized application
- Models are **not stored in Git**
- Models are mounted at runtime using Docker volumes

---

## ⚙️ Model Loading Strategy

The application supports two inference strategies:

- **Lazy loading** (default):  
  Models load on the first request.

- **Preloaded models**:  
  Models load during application startup using FastAPI lifespan hooks.

This behavior is controlled through configuration, making the app flexible for local development and production environments.

---

## 🐳 Running the Application (Docker)
### Prerequisites
- Docker installed and running
- Trained models available locally (mounted at runtime)
### 1️⃣ Build the Docker image

```docker build -t movie4u .```
### 2️⃣ Run the container with models mounted
```docker run -p 8000:8000 \-v "<ABSOLUTE_PATH_TO_MODELS>:/app/models" movie4u```
- Example
```docker run -p 8000:8000 -v "D:\Projects\Smart-text-engine\models:/app/models" movie4u```
### 3️⃣ Open in browser
```http://localhost:8000```

### 📚 Key Learnings from Iteration 1

- ML models should not be committed to Git repositories

- Inference code must be decoupled from training notebooks

- Dockerizing ML applications introduces new challenges (paths, resources, NLP assets)

- Lazy vs preloaded model loading is a meaningful performance trade-off

- NLP libraries often rely on external resources that must be handled explicitly in containers
### 🔮 What’s Next (Iteration 2)

#### Planned improvements include:

- Basic movie recommendation logic

- Improved UI and user experience

- Better caching of NLP resources

- Deployment to a cloud environment

- Performance and observability improvements
### 🧠 Final Note

- This repository represents Iteration 1 of a larger system.
The focus here was correctness, structure, and learning — not feature completeness.

- Future iterations will build on this foundation.