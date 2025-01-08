# Document Summary Generator and Q&A

This project is a desktop application built with PySide6. It allows users to upload documents (PDFs), summarize them, and ask questions related to the content. The application leverages pre-trained models from Hugging Face Transformers for summarization and question-answering tasks.

## Features

### Document Upload and Summarization
- Upload a PDF document.
- Generate a summary using advanced pre-trained NLP models.

### Visual Q&A
- Ask questions related to the document content.
- Receive answers based on the document context.

### User Interface
- Clean and intuitive UI built with PySide6.
- Upload and clear documents easily.
- Generate and view summaries in a dedicated section.
- Question input and response displayed in real-time.

### Advanced Functionality
- Manages session context for document-specific interactions.
- Provides a scrollable view for multi-page PDFs.

---

## User Interface Overview

### Layout
- **Top Section**: Application title.
- **Left Panel**: Document viewer and generated summary.
- **Right Panel**: Q&A area for input and responses.
- **Bottom Section**: Control buttons for uploading, clearing, and generating content.

### Buttons
- `Upload Document`: Upload a PDF for processing.
- `Generate`: Generate a summary or an answer.
- `Clear Document`: Clear the uploaded document and session context.
- `Clear All`: Clear all input, output, and session data.

---

## Code Overview

### Main Components
- **MainWindow**: The primary application class, managing the UI and functionality.
- **Threads**: Separate threads for tasks to ensure a smooth user experience.
  - `PDFLoaderThread`: Loads and processes PDF documents.
  - `SummarizationThread`: Generates summaries using a pre-trained summarization model.
  - `QAThread`: Answers user questions based on document content.

### Styling
- Custom button styles for normal, pressed, and disabled states.
- Default styles for text areas and scrollable containers.

---
![Application Interface](https://github.com/MarinosAntoniouCs/DocumentSummarizerQ-A/blob/main/DocumentSummarizerInterface.png)

## How to Run the Application

### Prerequisites
- Python 3.8+ installed.
- Required dependencies listed in `requirements.txt`.

### Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/document-summary-qa.git
   cd document-summary-qa
