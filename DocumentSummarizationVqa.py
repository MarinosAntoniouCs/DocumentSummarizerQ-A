import sys
import torch
from PIL import Image, ImageQt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QLabel, QTextEdit, QPushButton, QWidget, QFrame, QFileDialog, 
    QSizePolicy, QScrollArea
)
from PySide6.QtGui import QFont, QPixmap, QIcon
from PySide6.QtCore import Qt, QThread, Signal
from transformers import LEDForConditionalGeneration, LEDTokenizer
import fitz  # PyMuPDF
from huggingface_hub import login
import os
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, SimpleDirectoryReader, ChatPromptTemplate
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from dotenv import load_dotenv
import shutil

# Load environment variables
load_dotenv()

# Configure the Llama index settings
Settings.llm = HuggingFaceInferenceAPI(
    model_name="google/gemma-1.1-7b-it",
    tokenizer_name="google/gemma-1.1-7b-it",
    context_window=3000,
    token="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",
    max_new_tokens=512,
    generate_kwargs={"temperature": 0.1},
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Define the directory for persistent storage and data
PERSIST_DIR = "./db"
DATA_DIR = "data"

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

HUGGINGFACE_TOKEN = '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'
login(HUGGINGFACE_TOKEN)

# Styles
BUTTON_STYLE_NORMAL = """
    background-color: #007BFF; 
    color: black; 
    border: 2px solid #555; 
    border-radius: 20px; 
    padding: 5px; 
    font-family: 'Arial'; 
    font-size: 14px; 
    font-weight: bold;
"""

BUTTON_STYLE_PRESSED = """
    background-color: #0056b3; 
    color: white; 
    border: 2px solid #555; 
    border-radius: 20px; 
    padding: 5px; 
    font-family: 'Arial'; 
    font-size: 14px; 
    font-weight: bold;
"""

DISABLED_STYLE = """
    background-color: #b3d7ff; 
    color: grey; 
    border: 2px solid #555; 
    border-radius: 20px; 
    padding: 5px; 
    font-family: 'Arial'; 
    font-size: 14px; 
    font-weight: bold;
"""

STATUS_STYLE = """
    background-color: #f0f0f0;
    color: grey;
    font-style: italic;
"""

def clear_storage():
    # Clear the persistent storage directory
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
    os.makedirs(PERSIST_DIR, exist_ok=True)

    # Additionally, clear any in-memory storage or reset variables related to indexing if needed
    global index  # Ensure global reference to index is cleared
    index = None


def data_ingestion():
    clear_storage()
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)


def handle_query(query, context):
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    chat_text_qa_msgs = [
        (
            "user",
            f"""You are a Q&A assistant named CHATTO, created by Suriya. You have a specific response programmed for when users specifically ask about your creator, Suriya. The response is: "I was created by Suriya, an enthusiast in Artificial Intelligence. He is dedicated to solving complex problems and delivering innovative solutions. With a strong focus on machine learning, deep learning, Python, generative AI, NLP, and computer vision, Suriya is passionate about pushing the boundaries of AI to explore new possibilities." For all other inquiries, your main goal is to provide answers as accurately as possible, based on the instructions and context you have been given. If a question does not match the provided context or is outside the scope of the document, kindly advise the user to ask questions within the context of the document.
            Context:
            {context if context else "{context_str}"}
            Question:
            {query}
            """
        )
    ]
    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)
    
    query_engine = index.as_query_engine(text_qa_template=text_qa_template)
    answer = query_engine.query(query)
    
    if hasattr(answer, 'response'):
        return answer.response
    elif isinstance(answer, dict) and 'response' in answer:
        return answer['response']
    else:
        return "Sorry, I couldn't find an answer."


class PDFLoaderThread(QThread):
        pdf_loaded = Signal(list)  # Signal to emit when PDF is loaded

        def __init__(self, file_path, max_width=800, max_height=2500):
            super().__init__()
            self.file_path = file_path
            self.max_width = max_width
            self.max_height = max_height

        def run(self):
            doc = fitz.open(self.file_path)
            images = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Resize the image to fit the max width while maintaining aspect ratio
                width_ratio = self.max_width / float(img.size[0])
                height_ratio = self.max_width / float(img.size[1])
                scaling_factor = min(width_ratio, height_ratio)

                new_width = int(img.size[0] * scaling_factor)
                new_height = int(img.size[1] * scaling_factor)
                img = img.resize((new_width, new_height), Image.LANCZOS)

                qt_img = ImageQt.ImageQt(img)
                pixmap = QPixmap.fromImage(qt_img)
                images.append(pixmap)

            self.pdf_loaded.emit(images)


class SummarizationModelLoader(QThread):
    model_loaded = Signal(object, object)

    def run(self):
        tokenizer = LEDTokenizer.from_pretrained("patrickvonplaten/led-large-16384-pubmed")
        model = LEDForConditionalGeneration.from_pretrained("patrickvonplaten/led-large-16384-pubmed").to("cpu")
        self.model_loaded.emit(tokenizer, model)


class PDFSummarizer(QThread):
    summary_generated = Signal(str)

    def __init__(self, tokenizer, model, file_path):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.file_path = file_path

    def run(self):
        doc = fitz.open(self.file_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()

        inputs = self.tokenizer(text, return_tensors="pt", max_length=16384, truncation=True).input_ids.to("cpu")
        global_attention_mask = torch.zeros_like(inputs)
        global_attention_mask[:, 0] = 1

        summary_ids = self.model.generate(inputs, global_attention_mask=global_attention_mask)
        summary = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]
        self.summary_generated.emit(summary)


class SummarizationThread(QThread):
    summary_generated = Signal(str)

    def __init__(self, tokenizer, model, file_path):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.file_path = file_path

    def run(self):
        doc = fitz.open(self.file_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()

        inputs = self.tokenizer(text, return_tensors="pt", max_length=16384, truncation=True).input_ids.to("cpu")
        global_attention_mask = torch.zeros_like(inputs)
        global_attention_mask[:, 0] = 1

        summary_ids = self.model.generate(inputs, global_attention_mask=global_attention_mask)
        summary = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]
        self.summary_generated.emit(summary)


class QAThread(QThread):
    answer_generated = Signal(str)

    def __init__(self, question, context):
        super().__init__()
        self.question = question
        self.context = context  # Pass the current session context

    def run(self):
        # Pass the context to the query handler if needed
        answer = handle_query(self.question, self.context)
        self.answer_generated.emit(answer)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Document Summary Generator and Q&A")
        self.setWindowIcon(QIcon("documentIcon.png"))

        # Initialize UI elements
        self.init_ui()

        # Placeholder for models
        self.tokenizer = None
        self.model = None
        self.qa_tokenizer = None
        self.qa_model = None
        self.qa_chain = None

        # Flags to track model loading
        self.model_loaded = False
        self.qa_model_loaded = False

        # Track current session state
        self.current_session_context = None  # Store context for the current session
        
        self.qa_thread = None  # Initialize qa_thread here

        # Load models using threads
        self.load_models()
        
    def clear_storage_and_context(self):
        # Clear the storage
        clear_storage()

        # Reset the session context
        self.current_session_context = None

        # Clear the output areas if needed
        self.summary_area.clear()
        self.output_area.clear()

    def load_models(self):
        self.summary_loader = SummarizationModelLoader()
        self.summary_loader.model_loaded.connect(self.on_model_loaded)
        self.summary_loader.start()
        
    def init_ui(self):
        self.document_uploaded = False  # Track whether a document is uploaded
        self.text_in_input_area = False  # Track whether there is text in the input area

        # Create the main layout
        main_layout = QVBoxLayout()

        # Create the top container for the title
        top_container = QFrame()
        top_container.setFrameShape(QFrame.Box)
        top_container.setFixedHeight(100)  # Increase height for symmetry
        top_layout = QVBoxLayout(top_container)
        top_container.setStyleSheet("background-color: #2c2c2c;")  # Grayish black background

        title_label = QLabel("Document Summary Generator and Q&A")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Times New Roman", 34, QFont.Bold))
        title_label.setStyleSheet("color: white;")  # White text color
        top_layout.addWidget(title_label)

        # Create the middle container
        middle_container = QFrame()
        middle_container.setFrameShape(QFrame.Box)
        middle_layout = QHBoxLayout(middle_container)
        middle_container.setStyleSheet("background-color: #2c2c2c;")  # Grayish black background

        # Create the left container for document and summary
        left_container = QFrame()
        left_container.setFrameShape(QFrame.Box)
        left_layout = QVBoxLayout(left_container)
        left_container.setStyleSheet("background-color: #2c2c2c;")  # Grayish black background

        # Split the left container horizontally
        upper_left_container = QFrame()
        upper_left_layout = QVBoxLayout(upper_left_container)
        upper_left_container.setStyleSheet("background-color: #2c2c2c;")  # Grayish black background

        # Add a scroll area for images
        self.image_scroll_area = QScrollArea()
        self.image_scroll_area.setWidgetResizable(True)
        self.image_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.image_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.image_container = QWidget()
        self.image_layout = QVBoxLayout(self.image_container)  # Changed to QVBoxLayout for vertical scrolling
        self.image_scroll_area.setWidget(self.image_container)
        self.image_scroll_area.setStyleSheet("""
            QScrollBar:horizontal, QScrollBar:vertical {
                border: 1px solid #999999;
                background: #f0f0f0;
            }

            QScrollBar::handle:horizontal, QScrollBar::handle:vertical {
                background: #cccccc;
                min-width: 20px;
                min-height: 20px;
            }

            QScrollBar::add-line:horizontal, QScrollBar::add-line:vertical {
                border: 1px solid #999999;
                background: #d3d3d3;
                width: 20px;
                subcontrol-position: right;
                subcontrol-origin: margin;
            }

            QScrollBar::sub-line:horizontal, QScrollBar::sub-line:vertical {
                border: 1px solid #999999;
                background: #d3d3d3;
                width: 20px;
                subcontrol-position: left;
                subcontrol-origin: margin;
            }

            QScrollBar:left-arrow:horizontal, QScrollBar::right-arrow:horizontal,
            QScrollBar:up-arrow:vertical, QScrollBar::down-arrow:vertical {
                width: 3px;
                height: 3px;
            }

            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal,
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)

        self.upload_button = QPushButton("Upload Document")
        self.upload_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.upload_button.setFixedSize(150, 40)  # Set fixed size for the button
        self.upload_button.setStyleSheet(BUTTON_STYLE_NORMAL)
        self.upload_button.setEnabled(True)  # Initialize as enabled
        self.upload_button.pressed.connect(lambda: self.upload_button.setStyleSheet(BUTTON_STYLE_PRESSED))
        self.upload_button.released.connect(lambda: self.upload_button.setStyleSheet(BUTTON_STYLE_NORMAL))

        self.clear_document_button = QPushButton("Clear Document")
        self.clear_document_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.clear_document_button.setFixedSize(150, 40)  # Set fixed size for the button
        self.clear_document_button.setEnabled(False)  # Disable the button initially
        self.clear_document_button.setStyleSheet(DISABLED_STYLE)
        self.clear_document_button.pressed.connect(lambda: self.clear_document_button.setStyleSheet(BUTTON_STYLE_PRESSED))
        self.clear_document_button.released.connect(lambda: self.clear_document_button.setStyleSheet(DISABLED_STYLE))

        # Updated button layout code...
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.upload_button)
        button_layout.addWidget(self.clear_document_button)
        button_layout.addStretch()

        upper_left_layout.addWidget(self.image_scroll_area)
        upper_left_layout.addLayout(button_layout)

        lower_left_container = QFrame()
        lower_left_layout = QVBoxLayout(lower_left_container)
        lower_left_container.setStyleSheet("background-color: #2c2c2c;")  # Grayish black background

        self.summary_area = QTextEdit()
        self.summary_area.setPlaceholderText("Summary generated:")
        self.summary_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)  # Always show vertical scroll bar
        self.summary_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Show horizontal scroll bar as needed
        self.summary_area.setStyleSheet(
            "border: 2px solid #cccccc; border-radius: 5px; padding: 5px; "
            "background-color: #f0f0f0; color: black;"
        )  # Light gray border, light gray background, black text for summary area
        lower_left_layout.addWidget(self.summary_area)

        left_layout.addWidget(upper_left_container)
        left_layout.addWidget(lower_left_container)

        # Ensure equal space distribution
        left_layout.setStretchFactor(upper_left_container, 1)
        left_layout.setStretchFactor(lower_left_container, 1)

        # Create the right container for input and output areas
        right_container = QFrame()
        right_container.setFrameShape(QFrame.Box)
        right_layout = QVBoxLayout(right_container)
        right_container.setStyleSheet("background-color: #2c2c2c;")  # Grayish black background

        self.input_area = QTextEdit()
        self.input_area.setPlaceholderText("Ask a question:")
        self.input_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)  # Always show vertical scroll bar
        self.input_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Show horizontal scroll bar as needed
        self.input_area.setStyleSheet(
            "border: 2px solid #cccccc; border-radius: 5px; padding: 5px; "
            "background-color: #f0f0f0; color: black;"
        )  # Light gray border, light gray background, black text for input area
        self.input_area.textChanged.connect(self.update_clear_button_state)  # Connect textChanged signal

        right_layout.addWidget(self.input_area)

       # Create the generate and clear buttons
        self.generate_button = QPushButton("Generate")
        self.generate_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.generate_button.setFixedSize(150, 40)
        self.generate_button.setStyleSheet(BUTTON_STYLE_NORMAL)
        self.generate_button.setEnabled(True)

        # Connect signals for visual feedback
        self.generate_button.pressed.connect(lambda: self.generate_button.setStyleSheet(BUTTON_STYLE_PRESSED))
        self.generate_button.released.connect(lambda: self.generate_button.setStyleSheet(BUTTON_STYLE_NORMAL))
        self.generate_button.setEnabled(True)

        self.clear_button = QPushButton("Clear")
        self.clear_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.clear_button.setFixedSize(150, 40)
        self.clear_button.setStyleSheet(DISABLED_STYLE)

        # Connect signals for visual feedback
        self.clear_button.pressed.connect(lambda: self.clear_button.setStyleSheet(BUTTON_STYLE_PRESSED))
        self.clear_button.released.connect(lambda: self.clear_button.setStyleSheet(BUTTON_STYLE_NORMAL))

        # Create a horizontal layout to center the buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.generate_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()

        right_layout.addLayout(button_layout)

        self.output_area = QTextEdit()
        self.output_area.setPlaceholderText("Answer:")
        self.output_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)  # Always show vertical scroll bar
        self.output_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Show horizontal scroll bar as needed
        self.output_area.setStyleSheet(
            "border: 2px solid #cccccc; border-radius: 5px; padding: 5px; "
            "background-color: #f0f0f0; color: black;"
        )  # Light gray border, light gray background, black text for output area
        right_layout.addWidget(self.output_area)

        middle_layout.addWidget(left_container)
        middle_layout.addWidget(right_container)

        # Add all containers to the main layout
        main_layout.addWidget(top_container)
        main_layout.addWidget(middle_container)

        # Set the central widget and layout
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Connect button actions
        self.upload_button.clicked.connect(self.upload_document)
        self.generate_button.clicked.connect(self.generate_summary_and_answer)
        self.clear_document_button.clicked.connect(self.clear_document)
        self.clear_button.clicked.connect(self.clear_all)


    def on_model_loaded(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        print("Summarization model loaded")

    def upload_document(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Upload Document", "", "Documents (*.pdf)")

        if file_path:
            # Clear previous data and context
            self.clear_images()  # Clear existing images
            self.clear_storage_and_context()  # Clear storage and reset session context

            # Load the new document
            self.pdf_loader_thread = PDFLoaderThread(file_path)
            self.pdf_loader_thread.pdf_loaded.connect(self.on_pdf_loaded)
            self.pdf_loader_thread.start()
            self.current_document_path = file_path
            self.document_uploaded = True
            self.update_clear_document_button_state()

            # Disable the generate button and show status message in summary area
            self.generate_button.setEnabled(False)
            self.generate_button.setStyleSheet(DISABLED_STYLE)
            self.summary_area.setText("Summary is being generated...")
            self.summary_area.setStyleSheet(STATUS_STYLE)

            # Disable the clear document button while generating summary
            self.clear_document_button.setEnabled(False)
            self.clear_document_button.setStyleSheet(DISABLED_STYLE)

            # Clear the output area when a new document is uploaded
            self.output_area.clear()
            self.output_area.setStyleSheet("background-color: #f0f0f0; color: black;")

            # Perform data ingestion for the new document
            data_ingestion()

            # Start the summarization process for the new document
            if self.tokenizer and self.model:
                self.summarization_thread = SummarizationThread(self.tokenizer, self.model, self.current_document_path)
                self.summarization_thread.summary_generated.connect(self.on_summary_generated)
                self.summarization_thread.start()
            else:
                self.output_area.setText("Summarization model is still loading, please wait...")
                self.output_area.setStyleSheet(STATUS_STYLE)


    def on_summary_generated(self, summary):
        self.summary_area.setText(summary)
        self.current_session_context = summary  # Store the summary as the session context
        self.summary_area.setStyleSheet("background-color: #f0f0f0; color: black;")  # Reset to default style
        self.generate_button.setEnabled(True)
        self.generate_button.setStyleSheet(BUTTON_STYLE_NORMAL)

        # Enable the clear document button after generating summary
        self.clear_document_button.setEnabled(True)
        self.clear_document_button.setStyleSheet(BUTTON_STYLE_NORMAL)

        # Reset the output area style to default
        self.output_area.clear()
        self.output_area.setStyleSheet("background-color: #f0f0f0; color: black;")  # Reset to default white background


    def generate_summary_and_answer(self):
        if not self.document_uploaded:
            self.output_area.setText("Please upload a document first.")
            self.output_area.setStyleSheet("background-color: #f0f0f0; color: black;")
            return

        if not self.tokenizer or not self.model:
            self.output_area.setText("Models are still loading, please wait.")
            self.output_area.setStyleSheet("background-color: #f0f0f0; color: black;")
            return

        question = self.input_area.toPlainText().strip()
        if question:
            if self.qa_thread and self.qa_thread.isRunning():
                self.qa_thread.quit()
                self.qa_thread.wait()

            # Append the question to the output area
            self.output_area.append(f"Q: {question}\n")

            # Maintain the session context by passing it to the query handler
            self.qa_thread = QAThread(question, self.current_session_context)
            self.qa_thread.answer_generated.connect(self.on_answer_generated)
            self.qa_thread.start()

            # Clear the input area after starting the QA thread
            self.input_area.clear()
        else:
            self.output_area.setText("")
            self.output_area.setStyleSheet("background-color: #f0f0f0; color: black;")


    def on_answer_generated(self, answer):
        # Append the answer to the output area
        self.output_area.append(f"A: {answer}\n")

        # Enable the clear button after an answer is generated
        self.clear_button.setEnabled(True)
        self.clear_button.setStyleSheet(BUTTON_STYLE_NORMAL)


    def clear_document(self):
        # Clear the document display area and summary area
        self.clear_images()
        self.summary_area.clear()

        # Reset the state variables
        self.document_uploaded = False
        self.current_document_path = None  # Clear the current document path

        # Clear the storage and reset the context
        self.clear_storage_and_context()

        # Update button states
        self.update_clear_document_button_state()
        self.update_clear_button_state()

        # Disable the generate button
        self.generate_button.setEnabled(False)
        self.generate_button.setStyleSheet(DISABLED_STYLE)

        # Clear the input and output areas
        self.input_area.clear()
        self.output_area.clear()


    def clear_all(self):
        # Clear the input and output areas
        self.input_area.clear()
        self.output_area.clear()
        self.current_session_context = None  # Reset the session context when clearing all

        # Update the button state after clearing
        self.update_clear_button_state()

        # Update button style to show it as pressed
        self.clear_button.setStyleSheet(BUTTON_STYLE_PRESSED)

        # Restore button style after the action is completed
        self.clear_button.setStyleSheet(BUTTON_STYLE_NORMAL)


    def update_clear_document_button_state(self):
        if self.document_uploaded:
            self.clear_document_button.setEnabled(True)
            self.clear_document_button.setStyleSheet(BUTTON_STYLE_NORMAL)
        else:
            self.clear_document_button.setEnabled(False)
            self.clear_document_button.setStyleSheet(DISABLED_STYLE)


    def update_clear_button_state(self):
        if self.input_area.toPlainText().strip() or self.output_area.toPlainText().strip():  # Check if there is text in the input/output areas
            self.clear_button.setEnabled(True)
            self.clear_button.setStyleSheet(BUTTON_STYLE_NORMAL)
        else:
            self.clear_button.setEnabled(False)
            self.clear_button.setStyleSheet(DISABLED_STYLE)


    def add_image_label(self, pixmap):
        image_label = QLabel()
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.image_layout.addWidget(image_label, 0, Qt.AlignCenter)

    def clear_images(self):
        while self.image_layout.count():
            child = self.image_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def on_pdf_loaded(self, images):
        for pixmap in images:
            self.add_image_label(pixmap)

    def closeEvent(self, event):
        # Wait for threads to finish before closing
        if self.summary_loader and self.summary_loader.isRunning():
            self.summary_loader.quit()
            self.summary_loader.wait()

        if self.summarization_thread and self.summarization_thread.isRunning():
            self.summarization_thread.quit()
            self.summarization_thread.wait()

        if self.pdf_loader_thread and self.pdf_loader_thread.isRunning():
            self.pdf_loader_thread.quit()
            self.pdf_loader_thread.wait()

        if self.qa_thread and self.qa_thread.isRunning():
            self.qa_thread.quit()
            self.qa_thread.wait()

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1200, 800)  # Adjust window size as needed
    window.show()
    sys.exit(app.exec())
