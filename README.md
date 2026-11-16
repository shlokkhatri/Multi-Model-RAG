# PDF-Based Multimodal RAG System

A PDF-based **Multimodal Retrieval-Augmented Generation (RAG)** system that extracts, processes, retrieves, and understands information from text, tables, charts, and images in PDF documents.

The system combines document processing, visual understanding, semantic embeddings, vector search, and evidence-based answer generation to provide answers grounded in the original PDF content.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
- [Project Workflow](#project-workflow)
- [Information Extraction](#information-extraction)
- [Chunking Strategy](#chunking-strategy)
- [Embedding Generation](#embedding-generation)
- [Vector Database](#vector-database)
- [Retrieval and Answer Generation](#retrieval-and-answer-generation)
- [Evidence Traceability](#evidence-traceability)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Environment Configuration](#environment-configuration)
- [Running the Project](#running-the-project)
- [Demonstration](#demonstration)
- [Example Query](#example-query)
- [Future Improvements](#future-improvements)

---

## Project Overview

Traditional RAG systems primarily work with text-based documents. However, PDF documents frequently contain important information in tables, charts, graphs, and images.

This project addresses that limitation by implementing a multimodal PDF processing pipeline.

It extracts different types of information from PDF documents, converts their textual representations into vector embeddings, stores them in PostgreSQL using pgvector, and retrieves relevant evidence when a user asks a question.

The retrieved information is then passed to Google's Gemini model to generate evidence-grounded answers.

### Primary Objective

To develop a RAG system capable of understanding and answering questions using multiple information formats within PDF documents rather than relying only on extracted text.

---

## Key Features

- **PDF Text Extraction:** Extracts textual content from PDF pages.
- **Table Extraction:** Extracts structured table information, including rows, columns, and values.
- **Chart and Visual Extraction:** Identifies and extracts visual elements, including vector-based PDF graphics.
- **Visual Understanding:** Generates textual descriptions of charts and other visual elements.
- **Original Image Preservation:** Retains extracted visual images for multimodal model analysis.
- **Intelligent Chunking:** Divides extracted information into smaller retrieval units while preserving metadata.
- **Semantic Embeddings:** Generates 1536-dimensional embeddings using Google's embedding model.
- **Vector Search:** Retrieves semantically relevant document chunks using PostgreSQL and pgvector.
- **Multimodal Context Building:** Combines text, tables, visual descriptions, and original images.
- **Evidence-Based Generation:** Generates claims linked to retrieved evidence IDs.
- **Evidence Validation:** Checks whether generated claims reference valid retrieved evidence.
- **Semantic Verification:** Uses Gemini to evaluate whether generated claims are supported by evidence.
- **Evidence Precision Classification:** Distinguishes exact table values from textual information and approximate visual estimates.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.10 | Main programming language |
| PyMuPDF | PDF reading and content extraction |
| Google Gemini API | Visual understanding, answer generation, and semantic verification |
| Gemini 2.5 Flash | Answer generation and verification |
| gemini-embedding-001 | Text embedding generation |
| PostgreSQL 18 | Database |
| pgvector 0.8.7 | Vector storage and similarity search |
| python-dotenv | Environment variable management |
| VS Code | Development environment |
| PowerShell | Project execution and testing |

---

## System Architecture

The project follows a document ingestion pipeline followed by a question-answering pipeline.

### 1. Document Ingestion

The PDF is processed to extract:

- Text
- Tables
- Embedded images
- Vector-based visual elements

### 2. Content Processing

Extracted information is converted into structured representations.

- Text is divided into smaller chunks.
- Tables are represented with their structured information.
- Visual elements are converted into textual descriptions.
- Original images are preserved.

### 3. Chunking

The extracted information is organized into smaller retrieval units with associated metadata.

### 4. Embedding Generation

The textual representations of chunks are converted into numerical vectors using Google's embedding model.

### 5. Vector Storage

Chunks, embeddings, and metadata are stored in PostgreSQL using pgvector.

### 6. Query Processing

A user's question is converted into an embedding and used to retrieve relevant chunks.

### 7. Multimodal Context Construction

Retrieved text, tables, visual descriptions, and original images are combined into a context.

### 8. Answer Generation and Verification

Gemini generates claims linked to evidence IDs. These IDs are validated, and a semantic verification stage checks whether the claims are supported.

### 9. Final Answer

The system generates an answer with references to the retrieved document evidence.

---

## Project Workflow

```text
                 PDF DOCUMENT
                      |
                      v
              PDF INGESTION
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
        TEXT        TABLES      VISUALS
       EXTRACTION  EXTRACTION  EXTRACTION
          |           |           |
          |           |           v
          |           |     VISUAL UNDERSTANDING
          |           |           |
          |           |     VISUAL DESCRIPTIONS
          |           |           |
          +-----------+-----------+
                      |
                      v
                CHUNKING
                      |
                      v
              EMBEDDING GENERATION
                      |
                      v
             POSTGRESQL + PGVECTOR
                      |
                      |
               USER QUESTION
                      |
                      v
              QUERY EMBEDDING
                      |
                      v
              VECTOR RETRIEVAL
                      |
                      v
            MULTIMODAL CONTEXT
                      |
                      v
              GEMINI GENERATION
                      |
                      v
              EVIDENCE VALIDATION
                      |
                      v
              SEMANTIC VERIFICATION
                      |
                      v
                FINAL ANSWER
```

---

## Information Extraction

### 1. Text Extraction

PyMuPDF is used to read PDF pages and extract textual content.

The extracted content includes:

- Headings
- Paragraphs
- Explanatory text
- Captions
- Other textual information

Page numbers and document information are preserved to support source traceability.

### 2. Table Extraction

Tables are processed separately to preserve their structured information.

The system extracts information such as:

- Table titles
- Column headings
- Row labels
- Numerical values
- Units

For example, the source document contains a table titled:

**Table 2. Installed capacity and projected growth**

| Technology | 2025 (GW) | 2030E (GW) |
|---|---:|---:|
| Solar PV | 2,000 | 3,880 |
| Wind | 1,260 | 2,030 |
| Hydro | 1,425 | 1,500 |
| Bioenergy | 176 | 216 |
| **Total** | **4,861** | **7,626** |

The structured representation helps the system answer numerical questions accurately.

### 3. Chart and Visual Extraction

PDF charts are not always stored as ordinary embedded images. Some are created using vector drawing instructions.

Therefore, the project includes visual extraction that handles relevant vector-based visual elements in addition to embedded images.

The extracted visuals are saved as image files.

### 4. Visual Understanding

After extracting a chart or visual, a visual-understanding process generates a textual description.

The description can include:

- Chart title
- Chart type
- X-axis and Y-axis
- Units
- Legend
- Trends
- Approximate numerical values

The original image is also preserved.

This allows the system to use the textual description during semantic retrieval and the original image during multimodal analysis.

**Important:** Values estimated from charts are treated as approximate. Exact numerical answers should preferentially use structured table evidence.

---

## Chunking Strategy

Chunking divides extracted document information into smaller units so that relevant content can be retrieved efficiently.

The project uses the following configuration in:

`app/processing/chunker.py`

```python
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 500
```

These settings represent character-based chunking parameters.

### Why Overlap Is Used

When text is divided into chunks, important context may otherwise be lost at chunk boundaries.

Overlap preserves part of the previous chunk in the next chunk.

This helps maintain continuity between related sentences and paragraphs.

### Chunk Distribution

The processed 22-page PDF generated 52 chunks.

| Chunk Type | Number of Chunks |
|---|---:|
| Text | 28 |
| Table | 7 |
| Visual / Image | 17 |
| **Total** | **52** |

### Metadata Preservation

Each chunk can retain metadata such as:

- Chunk ID
- Document name
- Page number
- Content type
- Image path
- Additional extraction information

This metadata supports retrieval and evidence traceability.

Tables and visual descriptions are handled separately from ordinary text to preserve their context and structure.

---

## Embedding Generation

After chunking, the system converts textual chunk representations into numerical vectors.

The project uses:

**Model:** `gemini-embedding-001`

**Embedding Dimension:** `1536`

An embedding represents semantic information numerically.

Chunks with similar meanings can be retrieved based on their vector similarity.

### Embedding Process

1. Read the unified chunk collection.
2. Prepare the textual representation of each chunk.
3. Send the content to the Gemini embedding model.
4. Generate a 1536-dimensional vector.
5. Associate the vector with its original chunk and metadata.
6. Save embedding progress using a checkpoint.

The original image pixels are not assumed to be embedded by this text-embedding step. Visual descriptions participate in vector retrieval, while original images can be supplied separately to multimodal Gemini requests.

### Embedding Checkpoint

The project uses:

`embedding_checkpoint.json`

This helps track completed embedding operations and avoid unnecessarily processing already embedded chunks.

---

## Vector Database

The project uses PostgreSQL with pgvector for vector storage and similarity search.

### Database Configuration

| Component | Configuration |
|---|---|
| Database | `multimodal_rag` |
| Database Server | PostgreSQL 18 |
| Vector Extension | pgvector 0.8.7 |
| Embedding Dimension | 1536 |

### Main Table

The project stores document information in:

`document_chunks`

Important columns include:

| Column | Description |
|---|---|
| chunk_id | Unique chunk identifier |
| document_name | Source PDF name |
| page_number | Original PDF page |
| content_type | Text, table, or visual |
| content | Extracted chunk content |
| image_path | Associated visual image path |
| embedding | Vector embedding |
| metadata | Additional metadata in JSONB format |
| created_at | Record creation timestamp |

Vector indexes support efficient similarity retrieval.

---

## Retrieval and Answer Generation

When a user asks a question, the system follows these steps.

### Step 1: Query Embedding

The user's question is converted into a vector using the same embedding model and dimensionality used for document chunks.

### Step 2: Vector Retrieval

The query embedding is compared with stored document embeddings.

The most relevant chunks are retrieved from PostgreSQL using pgvector.

### Step 3: Multimodal Context Building

The retrieved chunks are passed to the context builder.

The context may include:

- Relevant text
- Structured table information
- Visual descriptions
- Original visual images

### Step 4: Evidence Registry

Each retrieved chunk is registered with a unique evidence ID.

Example:

```text
table_page_5_2
visual_4_page_4_visual_2_1
text_page_3_1
```

These IDs allow generated claims to reference their supporting evidence.

### Step 5: Claim Generation

Gemini generates structured claims associated with evidence IDs.

This makes it possible to identify which retrieved chunks support individual statements.

### Step 6: Evidence ID Validation

The system checks whether the generated evidence IDs exist in the retrieved evidence registry.

Invalid references must not be accepted.

### Step 7: Semantic Verification

A separate verification stage evaluates whether the generated claims are supported by the associated evidence.

The system follows fail-closed behavior: if verification fails or cannot be completed, unverified claims should not be accepted.

### Step 8: Final Answer

The final response contains the generated answer and source references.

---

## Evidence Traceability

An important part of this project is identifying the source of generated information.

The evidence registry includes precision classifications.

| Content Type | Precision Classification | Priority |
|---|---|---:|
| Table | `exact_structured` | 1 |
| Text | `textual` | 2 |
| Visual / Image | `approximate_visual` | 3 |
| Unknown | `unknown` | 4 |

### Why Precision Classification Matters

Consider a question about projected renewable capacity in 2030.

A table provides an exact value:

```text
Total renewable capacity = 7,626 GW
```

A chart may visually indicate an approximate value:

```text
Approximate chart estimate = 7,900 GW
```

These values should not automatically be treated as contradictory.

The system should consider:

- Source precision
- Year
- Units
- Scope
- Original chart and table content

Exact structured table values should be distinguished from approximate visual estimates.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
```

Navigate into the project:

```bash
cd Multimodel_rag
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Environment

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

If the project contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

If it does not, install the dependencies specified by the project's imports and dependency configuration.

### 5. Configure PostgreSQL

Ensure PostgreSQL is running and the required database exists.

Create the database if necessary:

```sql
CREATE DATABASE multimodal_rag;
```

Connect to the database and enable pgvector:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Ensure the `document_chunks` table and its indexes are created using the project's database initialization code.

### 6. Configure Gemini API

Create a Google Gemini API key and configure it in the environment file.

---

## Environment Configuration

Create a `.env` file in the project root.

```env
GEMINI_API_KEY=your_gemini_api_key

DB_HOST=localhost
DB_PORT=5432
DB_NAME=multimodal_rag
DB_USER=your_database_username
DB_PASSWORD=your_database_password
```

The database variable names above are illustrative. Match them to the actual names expected by your database connection module.

**Security:** Never commit `.env` files, API keys, or database passwords to GitHub.

Add the following to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
*.pyc
```

---

## Running the Project

Open PowerShell in the project root.

### Activate the Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Run the Answer Generation Module

```powershell
python -m app.generation.answer_generator
```

This executes the question-answering workflow implemented in the answer generator.

The current test query is:

```text
What are the global renewable energy capacity projections for 2030?
```

The ingestion, chunking, and embedding modules are separate preparation stages. They should be executed using their respective project scripts when processing a new PDF or rebuilding the processed data.

---

## Demonstration

### Source Document

**Global Renewable Energy Outlook**

The demonstration uses extracted text, tables, and a chart from the source PDF.

### Test Query 1

```text
What are the global renewable energy capacity projections for 2030?
```

### Retrieved Evidence

| Evidence ID | Evidence Type | Source |
|---|---|---|
| `table_page_5_2` | Table | Page 5 |
| `visual_4_page_4_visual_2_1` | Visual | Page 4 |
| `text_page_3_1` | Text | Page 3 |

### Generated Answer

The system generated the following claims:

1. Total global renewable energy capacity is projected to reach **7,626 GW** by 2030.
2. Solar PV capacity is projected to be **3,880 GW**.
3. Wind capacity is projected to be **2,030 GW**.
4. Hydro capacity is projected to be **1,500 GW**.
5. Bioenergy capacity is projected to be **216 GW**.
6. Chart evidence indicates a strong increasing trend from 2015 to 2030, with Solar PV becoming the largest single contributor.
7. Text evidence indicates that renewable capacity is projected to roughly double between 2025 and 2030, led by Solar PV.

The generated claims include references to their retrieved evidence IDs.

### Test Query 2

A second recommended test focuses on chart understanding:

```text
How does global renewable energy capacity change from 2015 to 2030,
and which technology contributes most to the projected growth?
```

Expected behavior:

- Retrieve the relevant chart description.
- Attach the original chart image.
- Identify the overall increasing trend.
- Identify Solar PV as a major contributor to projected growth.
- Use the table for exact numerical values if the answer requires them.

This is a proposed additional test query; it should be executed separately to record its actual output.

---

## Future Improvements

Potential improvements to the project include:

- More precise table extraction with row-level and cell-level references.
- Claim-level citations containing table title, row, column, and exact cell value.
- Chart citations containing chart title, axes, units, and approximate values.
- Better handling of conflicting values across different PDF sources.
- More robust Gemini retry and checkpoint recovery.
- Compact verification responses to reduce token truncation.
- Consistent verification statuses such as `SUPPORTED`, `UNSUPPORTED`, `UNCERTAIN`, and `CONFLICTED`.
- Regression testing across text-only, table-only, visual-only, mixed-evidence, conflicting-value, and not-in-document questions.
- Support for processing multiple PDF documents.

---

## Conclusion

This project demonstrates how multimodal RAG can be applied to PDF documents containing different types of information.

By combining PDF extraction, table processing, visual understanding, chunking, embeddings, PostgreSQL/pgvector, multimodal context construction, and evidence-based generation, the system goes beyond conventional text-only RAG.

Its main objective is to generate answers that are not only relevant but also traceable to the original document evidence.
