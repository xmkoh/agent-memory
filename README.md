# Agent Memory

A persistent, semantic memory layer for AI agents, built on **Weaviate**. It gives agents a structured file-system-like namespace plus vector-powered semantic search, all backed by a local Weaviate instance.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Connect](#1-connect)
  - [Write a Document](#2-write-a-document)
  - [List Folders](#3-list-folders)
  - [Semantic Search](#4-semantic-search)
  - [Read a Full File](#5-read-a-full-file)
  - [Chat / Conversational Memory](#6-chat--conversational-memory)
- [API Reference](#api-reference)
- [Schema Design](#schema-design)
- [LangChain Integration](#langchain-integration)

---

## Overview

`WeaviateMemoryManager` provides agents with two complementary memory modes:

| Mode | Collection | How it works |
|---|---|---|
| **Knowledge Base** | `Document` + `Chunk` | Markdown files are stored raw and split into heading-level chunks for semantic search |
| **Conversational** | `ChatMessage` | Per-thread chat history persisted as ordered messages |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Agent                            │
└───────────────────┬─────────────────────────────────┘
                    │ WeaviateMemoryManager
        ┌───────────┼───────────────┐
        ▼           ▼               ▼
  Document       Chunk          ChatMessage
  (raw text,   (vectorized     (thread history,
  no vector)    heading +       no vector)
                chunk_text)
        │           │
        └──hasDocument (cross-ref)──┘
```

- **`Document`** — stores the full raw Markdown. No vectorization (saves compute).
- **`Chunk`** — one record per `##` section, fully vectorized via `text2vec-transformers`. References back to its parent `Document`.
- **`ChatMessage`** — ordered by timestamp per `thread_id`. No vectors needed; looked up by exact match.

---

## Prerequisites

- [Docker](https://www.docker.com/) & Docker Compose
- Python 3.9+
- `weaviate-client` and `langchain-core` packages

```bash
pip install weaviate-client langchain-core
```

---

## Quick Start

### 1. Start Weaviate

A `docker-compose.yml` is included. It spins up Weaviate **1.28.2** alongside a local `sentence-transformers` embedding service:

```bash
docker compose up -d
```

- Weaviate REST/GraphQL → `http://localhost:8080`
- gRPC → `localhost:50051`
- Embedding model: `multi-qa-MiniLM-L6-cos-v1` (CPU, no CUDA required)

> Data is persisted in a named Docker volume (`weaviate_data`) across restarts.

---

## Usage

### 1. Connect

```python
import weaviate
from memory_manager import WeaviateMemoryManager

client = weaviate.connect_to_local()
memory = WeaviateMemoryManager(client)
```

`WeaviateMemoryManager.__init__` auto-creates the `Document`, `Chunk`, and `ChatMessage` collections if they don't already exist.

---

### 2. Write a Document

Store a Markdown file under a virtual folder path. The file is saved raw *and* split into semantic chunks automatically:

```python
memory.write_markdown(
    filename="project_kaya_architecture.md",
    folder_path="projects/kaya/team_a",
    content="""# Enterprise Platform Initialization
## Inference Engine
We will deploy the high-performance pipeline using Triton Inference Server.
## Workflow Sync
Dependency tracking between Team A and the vendor is critical for go-live.
"""
)
# → Successfully ingested project_kaya_architecture.md into 2 chunks.
```

Chunking splits on `##` headings. Each chunk is vectorized and linked back to the parent document via a cross-reference.

---

### 3. List Folders

Get all distinct virtual folder paths the agent has written to:

```python
folders = memory.list_folders()
# → ['projects/kaya/team_a']
```

List files inside a specific folder:

```python
files = memory.list_files_in_folder("projects/kaya/team_a")
# → ['project_kaya_architecture.md']
```

---

### 4. Semantic Search

Run a natural-language query across all vectorized chunks. Returns the top-3 most relevant results with source attribution:

```python
results = memory.search_knowledge_base(query="What inference engine are we using?")
print(results[0]["content"])
# → 'We will deploy the high-performance pipeline using Triton Inference Server.'
```

Each result contains:

```python
{
    "source_file": "projects/kaya/team_a/project_kaya_architecture.md",
    "heading": "Inference Engine",
    "content": "We will deploy the high-performance pipeline using Triton Inference Server."
}
```

Optionally restrict search to a single folder:

```python
results = memory.search_knowledge_base(
    query="go-live dependencies",
    restrict_to_folder="projects/kaya/team_a"
)
```

---

### 5. Read a Full File

Deterministic exact-match lookup — retrieves the full raw Markdown without any vector search:

```python
raw_text = memory.read_file_by_name("project_kaya_architecture.md")
```

Useful when an agent needs to rewrite or update an existing file.

---

### 6. Chat / Conversational Memory

Persist and retrieve multi-turn conversation history scoped by `thread_id`:

```python
# Write messages
memory.write_chat_message(thread_id="session-42", role="user", content="What engine are we using?")
memory.write_chat_message(thread_id="session-42", role="assistant", content="Triton Inference Server.")

# Read history (returned sorted by timestamp)
history = memory.read_chat_messages(thread_id="session-42")

# Clear a thread
memory.clear_chat_messages(thread_id="session-42")
```

---

## API Reference

### `WeaviateMemoryManager`

| Method | Description |
|---|---|
| `write_markdown(filename, folder_path, content)` | Ingest a Markdown string; saves raw + vectorized chunks |
| `read_file_by_name(filename)` | Exact-match file retrieval (full raw content) |
| `search_knowledge_base(query, restrict_to_folder?)` | Semantic vector search; returns top-3 chunks |
| `list_folders()` | Returns all distinct virtual folder paths |
| `list_files_in_folder(folder_path)` | Returns filenames inside a specific folder |
| `write_chat_message(thread_id, role, content, metadata?)` | Append a message to a conversation thread |
| `read_chat_messages(thread_id)` | Fetch sorted message history for a thread |
| `clear_chat_messages(thread_id)` | Delete all messages for a thread |

---

## Schema Design

| Collection | Vectorized | Key Properties |
|---|---|---|
| `Document` | ❌ | `filename`, `folder_path`, `raw_content`, `created_at` |
| `Chunk` | ✅ (`text2vec-transformers`) | `heading`, `chunk_text`, `hasDocument →` |
| `ChatMessage` | ❌ | `thread_id`, `role`, `content`, `timestamp`, `metadata` |

The **Parent-Child** pattern separates storage (raw file) from retrieval (semantic chunks), avoiding double-vectorization overhead.

---

## LangChain Integration

`WeaviateChatMessageHistory` implements `BaseChatMessageHistory` for drop-in use with any LangChain chain:

```python
from memory_manager import WeaviateChatMessageHistory

history = WeaviateChatMessageHistory(
    memory_manager=memory,
    thread_id="session-42"
)

# Works with RunnableWithMessageHistory, ConversationChain, etc.
```

Roles are mapped bidirectionally:

| Weaviate role | LangChain message type |
|---|---|
| `user` | `HumanMessage` |
| `assistant` | `AIMessage` |
| `system` | `SystemMessage` |
