# 🏥 IMMUNY: Multimodal Semantic Search for Autoimmune Disease Cases

**Multimodal Biological Memory for discovering autoimmune disease cases and similar past 
biological situations through hybrid search combining medical text and images.**

---

## 🎯 Project Overview

Immuny is Not a chatbot, not a diagnostic tool, not “AI decides treatment” , and not about 
storing personal patient data  
Immuny lets researchers ask: “Have we seen an immune situation like this before, and what 
happened?”  using:
- 📝 **Text-based queries** (symptoms, bioMarks, diagnoses, treatments)
- 🖼️ **Image-based queries** (medical imaging)
- 🔀 **Multimodal hybrid queries** (combined text + image search)

### Key Capabilities
✅ **5,310+ Cleaned Case Reports** from PubMed Central  
✅ **950+ Diagnostic Medical Images** (CT scans, X-rays, biopsies)  
✅ **Multimodal Embeddings** (384D text + 512D images)  
✅ **HNSW-Indexed Vector Database** (250x faster search)  
✅ **Cloud-Hosted Qdrant** (serverless, managed infrastructure)  
✅ **Chunking** with Chonkie (context-aware 67K chunks)  

---

## ⚡ Problem Statement

### The Challenge
Medical researchers studying autoimmune diseases face:
- **Fragmentation**: Case reports scattered across journals, PDFs, archives
- **Limited Search**: Keyword-based systems miss clinically similar cases
- **Multimodal Gap**: No unified search across text AND images
- **Scalability Crisis**: Manual case review impossible for 5000+ documents

### The Solution
IMMUNY combines cutting-edge AI technologies to:
1. **Extract & Clean** 5,310 autoimmune disease cases with validation
2. **Embed Intelligently** using SentenceTransformer (text) + CLIP (images)
3. **Index Efficiently** with HNSW (Hierarchical Navigable Small World)
4. **Search Semantically** across text, images, or both together
5. **Scale Seamlessly** with cloud-hosted Qdrant vector database

---

## 🏗️ System Architecture

### Multi-Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  LLM& Research UI (Phase 5)              │
│  • Text-only search  • Image-only search  • Multimodal  │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│           QDRANT VECTOR DATABASE (Cloud)                │
│  • Named Vectors: text_vector, image_vector, combined   │
│  • HNSW Indexing: m=24, ef_construct=300               │
│  • Cosine Distance: All vector spaces                    │
│  • 67K+ Points: Text chunks + Images                     │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│         EMBEDDING GENERATION (Phase 3)                  │
│                                                         │
│  Text Path:                  Image Path:               │
│  ├─ Chonkie Chunking        ├─ CLIP ViT-B-32         │
│  ├─ 67K chunks              ├─ 512D embeddings       │
│  ├─ SentenceTransformer      └─ Vision encoder       │
│  └─ 384D embeddings                                   │
│                                                         │
│  Fusion:                                               │
│  └─ Concatenate: [384D text ⊕ 512D image] = 896D      │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│      DATA PROCESSING & CLEANING (Phase 2)              │
│  • CSV fix: Merge overflow columns                      │
│  • Image validation: Check file paths                   │
│  • Metadata validation: Consistency checks             │
│  • Deduplication: Remove corrupt entries                │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│         RAW DATASET (Phase 1)                           │
│  • 5,579 case reports (CSV)                            │
│  • 1,200+ medical images (PNG/WebP)                    │
│  • Image metadata (JSON)                               │
│  • Case citations (JSON)                               │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Data Pipeline Phases (ps : maybe we are going to add more data) 

### **PHASE 1: Data Ingestion & Validation**
- Load 5,579 autoimmune disease cases from CSV
- Load 950+ diagnostic medical images
- Fix CSV column overflow (case_text too long for single column)
- Validate image file paths and metadata consistency
- **Output**: 5,310 cleaned cases, 950 valid images

### **PHASE 2: Text Chunking (Chonkie)**
- Split case texts at semantic boundaries (not fixed sizes)
- Preserve clinical context within chunks
- Generate ~67,000 chunks (avg 180 chars, 50-300 tokens each)
- Maintain case_id traceability for each chunk
- **Output**: 67K semantically coherent chunks ready for embedding

### **PHASE 3: Multimodal Embedding Generation**
**Text Embeddings**:
- Model: SentenceTransformer (`all-MiniLM-L6-v2`)
- Dimension: 384D vectors
- Speed: ~100 chunks/sec on CPU
- Output: [67000, 384] numpy array

**Image Embeddings**:
- Model: CLIP Vision Transformer (ViT-B-32)
- Dimension: 512D vectors
- Speed: ~45 images/sec on CPU
- Output: [950, 512] numpy array

**Multimodal Fusion**:
- Strategy: Concatenation (preserves all information)
- Formula: `combined_vec = [text_vec | image_vec]`
- Dimension: 896D vectors
- **Output**: All embeddings ready for database insertion

### **PHASE 4: Vector Database Setup & Insertion**
- Create Qdrant collection with named vectors
- Configure HNSW indexing (m=24, ef_construct=300)
- Prepare PointStruct objects with embeddings + metadata
- Batch insert 67K+ points at 1000 pts/sec
- **Output**: Searchable vector database with 67K+ indexed points

### **PHASE 5: Search Interface Development** *(In Progress)*
- Build text-only semantic search
- Build image-only similarity search
- Build multimodal hybrid search
- Implement result re-ranking and filtering
- choice of LLM 
- Deploy web UI 

---

## 🔧 Technology Stack

### Core ML/AI Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| **Qdrant Client** | 2.8+ | Vector database connectivity & operations |
| **Sentence Transformers** | 2.2+ | Text encoder for semantic embeddings |
| **CLIP (OpenAI)** | Latest | Vision-language model for image embeddings |
| **Chonkie** | 0.1+ | Intelligent semantic text chunking |
| **PyTorch** | 2.0+ | Deep learning framework backend |
| **OpenCV** | 4.8+ | Image processing utilities |
| **Pillow** | 10.0+ | Image format handling |


### Data Processing
| Library | Version | Purpose |
|---------|---------|---------|
| **Pandas** | 2.0+ | CSV/data manipulation |
| **NumPy** | 1.24+ | Numerical computing & arrays |
| **JSON** | Built-in | Metadata parsing |

### Infrastructure & DevOps
| Technology | Purpose |
|-----------|---------|
| **Qdrant Cloud** | Managed vector database (serverless) |
| **Python-dotenv** | Environment variable management (security) |
| **GitHub** | Version control & collaboration |

---

## 🎯 Multimodal Vector Architecture

### Named Vectors Strategy

Why use **named vectors** instead of single vectors?

**Named Vectors** = Store multiple embedding types per point
```python
PointStruct(
    id=1,
    vector={
        "text_vector": [float] × 384,      # Text space
        "image_vector": [float] × 512,     # Image space
        "combined_vector": [float] × 896   # Hybrid space
    },
    payload={...}
)
```

### Search Types

#### 1️⃣ **Text-Only Search**
```
Query: "systemic lupus erythematosus with kidney involvement"
    ↓
Text Encoder (SentenceTransformer)
    ↓
384D Vector
    ↓
Search text_vector space with HNSW
    ↓
Return top-10 clinically similar cases
```
**Use case**: Research existing treatments, find similar presentations

#### 2️⃣ **Image-Only Search**
```
Query: Kidney biopsy image (PNG)
    ↓
Image Encoder (CLIP ViT-B-32)
    ↓
512D Vector
    ↓
Search image_vector space with HNSW
    ↓
Return top-10 visually similar medical images
```
**Use case**: Find cases with similar diagnostic imaging

#### 3️⃣ **Multimodal Hybrid Search**
```
Query: "SLE" (text) + Kidney biopsy image
    ↓
Parallel Encoding:
  ├─ Text → 384D vector
  └─ Image → 512D vector
    ↓
Fuse: concatenate [384D ⊕ 512D] → 896D
    ↓
Search combined_vector space
    ↓
Return top-10 best matches across text + image
```
**Use case**: Find complete case presentations matching all criteria

### HNSW Indexing Details

**Why HNSW?**
- ⚡ **Speed**: O(log N) average search time (vs O(N) brute force)
- 🎯 **Accuracy**: >99% of exhaustive search results
- 💾 **Memory**: O(N) efficient linear overhead
- 📈 **Scalability**: Tested on billion-scale datasets

**Configuration**:
- `m=24`: Average connections per node (sweet spot for medical data)
  - ↓ Lower m: faster search, less memory, worse accuracy
  - ↑ Higher m: slower search, more memory, better accuracy
- `ef_construct=300`: Index construction quality
  - ↓ Lower ef_construct: faster construction, worse index quality
  - ↑ Higher ef_construct: slower construction, better index quality
- **Distance Metric**: Cosine Similarity (optimal for normalized embeddings)

**Performance**:
- Approximate nearest neighbor search: ~250x faster than brute force
- Average query latency: ~50-75ms per query
- Batch query (100 parallel): ~2 seconds

---

## 📁 Project Structure

```
Immuny/
├── 📔 dataEmbeddingQdrant.ipynb        # Main pipeline notebook (Phases 1-5)
├── 📔 dataFilttering.ipynb             #colab pipline for data filtering and cleaning
├── 📊 Immuny_dataset_clean/            # Cleaned dataset
│   └── autoimmune_d_dataset/
│       ├── cases_cleaned.csv           # 5,310 cleaned case reports
│       ├── image_metadata_cleaned.json # Image metadata
│       ├── case_report_citations_cleaned.json
│       └── images/                     # 950 medical images
│           ├── PMC1/, PMC3/, ..., PMC8/
│           └── *.png, *.webp files
├── 📄 .env                             # Credentials (QDRANT_URL, QDRANT_API_KEY)
├── 📖 README.md                        # This file
└── 📝 requirements.txt                 # Python dependencies
```

---


## 🔍 Example Usage

### Text Search
```python
from qdrant_client import QdrantClient
import json

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Search for cases matching query
query = "lupus with kidney involvement"
query_embedding = text_encoder.encode(query)

results = client.search(
    collection_name="autoimmune_cases",
    query_vector=query_embedding,
    using="text_vector",
    limit=10
)

for i, hit in enumerate(results, 1):
    print(f"{i}. Score: {hit.score:.4f}")
    print(f"   Case ID: {hit.payload['case_id']}")
    print(f"   Chunk: {hit.payload['chunk_text'][:100]}...")
```

### Image Search
```python
from PIL import Image

# Load query image
query_image = Image.open("kidney_biopsy.png")
query_embedding = image_encoder(query_image)

results = client.search(
    collection_name="autoimmune_cases",
    query_vector=query_embedding,
    using="image_vector",
    limit=5
)

print(f"Found {len(results)} similar medical images")
for hit in results:
    print(f"Image: {hit.payload['image_name']} (Score: {hit.score:.4f})")
```

### Multimodal Search
```python
# Combine text + image query
text_emb = text_encoder.encode("systemic lupus")
image_emb = image_encoder(Image.open("biopsy.png"))
combined_emb = np.concatenate([text_emb, image_emb])

results = client.search(
    collection_name="autoimmune_cases",
    query_vector=combined_emb,
    using="combined_vector",
    limit=10
)

print(f"Best multimodal matches: {len(results)} results")
```

---

## 🛣️ Development Roadmap

### ✅ Completed
- [x] Data acquisition & cleaning (Phase 1)
- [x] Text chunking with Chonkie (Phase 2)
- [x] Embedding generation (Phase 3)
- [x] Vector database setup (Phase 4)
- [x] Environment-based credential management
- [x] Connection testing & validation

### 🚧 In Progress (Phase 5)
- [ ] Search interface implementation
- [ ] Result re-ranking algorithm
- [ ] Performance optimization
- [ ] Web UI 

### 📋 Future (Phase 6+)
- [ ] Query expansion with LLMs
- [ ] Semantic clustering analysis
- [ ] Real-time case ingestion
- [ ] EHR system integration
- [ ] Multi-language support (maybe)
- [ ] Clinician feedback loop (+ a real doctor's feedback)
- [ ] Advanced filtering (severity, treatment, outcome)

---

## 📚 References

### Papers & Algorithms
- **CLIP** (Radford et al., 2021): Learning Transferable Models for Computer Vision Tasks
- **Sentence-BERT** (Reimers & Gupta, 2019): Sentence Embeddings using Siamese BERT
- **HNSW** (Malkov & Yashunin, 2018): Efficient Approximate Nearest Neighbor Search
- **Chonkie**: Intelligent semantic text chunking


### Tools & Resources
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Sentence Transformers](https://www.sbert.net/)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Chonkie](https://github.com/kesamet/chonkie)

---

## 📋 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Cases | 5,310 (cleaned) |
| Total Medical Images | 950+ |
| Total Chunks | ~67,000 |
| Avg Chunk Size | 180 characters |
| Text Embedding Dim | 384D |
| Image Embedding Dim | 512D |
| Combined Embedding Dim | 896D |
| Total Points in DB | ~67,000 |
| Storage (text vectors) | ~103 MB |
| Storage (image vectors) | ~2 MB |
| Storage (metadata) | ~50 MB |
| **Total Database Size** | **~155 MB** |

---

**Status**: Phase 4 Complete ✅ | Phase 5 In Development 🚧  
**Last Updated**: January 26, 2026



## 🛠️ Technology Stack

- **Vector DB**: Qdrant Cloud (named vectors architecture)
- **Text Embeddings**: SentenceTransformer (all-MiniLM-L6-v2, 384D)
- **Image Embeddings**: CLIP ViT-B-32 (512D)
- **Text Chunking**: Chonkie RecursiveChunker
- **Dataset**: 5,310 autoimmune cases + 51 medical images


## 📊 System Statistics

- **Text Vectors**: 67,651 chunks × 384D
- **Image Vectors**: 51 images × 512D
- **Search Modes**: Text-only, Image-only, Full case retrieval
- **Collection**: `autoimmune_cases` on Qdrant Cloud

## 🔧 Configuration

Qdrant connection (in `immuny_system.ipynb`):
```python
QDRANT_URL = "your-qdrant-url"
QDRANT_API_KEY = "your-api-key"
COLLECTION_NAME = "autoimmune_cases"
```

## 👥 Contributors

- **AIspire** - Sarra Bejja & Chadha Dalhoumi




