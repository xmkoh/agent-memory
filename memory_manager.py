import weaviate
from weaviate.classes.config import Configure, Property, DataType, ReferenceProperty
from weaviate.classes.query import Filter
from datetime import datetime, timezone
import uuid
import re
import fnmatch
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

class WeaviateMemoryManager:
    def __init__(self, client: weaviate.WeaviateClient):
        self.client = client
        self._initialize_schema()
        
        # Get references to the collections
        self.document_collection = self.client.collections.get("Document")
        self.chunk_collection = self.client.collections.get("Chunk")
        self.chat_collection = self.client.collections.get("ChatMessage")

    def _initialize_schema(self):
        """Sets up the Parent-Child architecture if it doesn't exist."""
        
        # 1. The Parent: Deterministic File Storage (No Vectors)
        if not self.client.collections.exists("Document"):
            self.client.collections.create(
                name="Document",
                properties=[
                    Property(name="filename", data_type=DataType.TEXT, skip_vectorization=True),
                    Property(name="folder_path", data_type=DataType.TEXT, skip_vectorization=True),
                    Property(name="raw_content", data_type=DataType.TEXT, skip_vectorization=True),
                    Property(name="created_at", data_type=DataType.DATE),
                ],
                # Turn off vectorization to save compute and storage for the raw file
                vectorizer_config=Configure.Vectorizer.none() 
            )

        # 2. The Child: Semantic Memory (Vectorized)
        if not self.client.collections.exists("Chunk"):
            self.client.collections.create(
                name="Chunk",
                properties=[
                    Property(name="heading", data_type=DataType.TEXT),
                    Property(name="chunk_text", data_type=DataType.TEXT),
                ],
                references=[
                    ReferenceProperty(name="hasDocument", target_collection="Document")
                ],
                # Configure your embedding model here (e.g., text2vec-openai, huggingface, etc.)
                vectorizer_config=Configure.Vectorizer.text2vec_transformers()
            )

        # 3. Conversational Memory (No vectors needed, exact thread_id lookup)
        if not self.client.collections.exists("ChatMessage"):
            self.client.collections.create(
                name="ChatMessage",
                properties=[
                    Property(name="thread_id", data_type=DataType.TEXT, skip_vectorization=True),
                    Property(name="role", data_type=DataType.TEXT, skip_vectorization=True),
                    Property(name="content", data_type=DataType.TEXT, skip_vectorization=True),
                    Property(name="timestamp", data_type=DataType.DATE),
                    Property(name="metadata", data_type=DataType.TEXT, skip_vectorization=True),
                ],
                vectorizer_config=Configure.Vectorizer.none()
            )

    def _chunk_content(self, content: str) -> list:
        """Naive Markdown chunking (splits by ## headers). Returns chunk dicts."""
        sections = re.split(r'(?m)^## ', content)
        intro = sections.pop(0) if sections else ""

        chunks = []
        if intro.strip():
            chunks.append({"heading": "Introduction", "chunk_text": intro.strip()})

        for section in sections:
            lines = section.split('\n', 1)
            heading = lines[0].strip()
            chunk_text = lines[1].strip() if len(lines) > 1 else ""
            if chunk_text:
                chunks.append({"heading": heading, "chunk_text": chunk_text})
        return chunks

    def _insert_chunks(self, chunks: list, doc_uuid):
        """Inserts vectorized chunks cross-referenced to a parent document."""
        for chunk in chunks:
            self.chunk_collection.data.insert(
                properties=chunk,
                references={"hasDocument": doc_uuid}
            )

    def _delete_chunks_for(self, doc_uuid):
        """Removes all chunks referencing a given parent document."""
        self.chunk_collection.data.delete_many(
            where=Filter.by_ref("hasDocument").by_id().equal(doc_uuid)
        )

    def get_document(self, filename: str):
        """Returns the raw Weaviate object for a file (uuid + properties), or None."""
        response = self.document_collection.query.fetch_objects(
            filters=Filter.by_property("filename").equal(filename),
            limit=1
        )
        return response.objects[0] if response.objects else None

    def write_markdown(self, filename: str, folder_path: str, content: str, overwrite: bool = False):
        """Ingests a file, saves the raw text, and streams vectorized chunks.

        If overwrite=True and a file with the same name exists, the existing
        document and its chunks are replaced rather than duplicated.
        """
        existing = self.get_document(filename)
        if existing is not None:
            if not overwrite:
                raise FileExistsError(
                    f"File '{filename}' already exists. Use edit_file or overwrite to modify it."
                )
            self._delete_chunks_for(existing.uuid)
            self.document_collection.data.delete_by_id(existing.uuid)

        # 1. Save the Parent Document
        doc_uuid = uuid.uuid4()
        self.document_collection.data.insert(
            uuid=doc_uuid,
            properties={
                "filename": filename,
                "folder_path": folder_path,
                "raw_content": content,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        )

        # 2 & 3. Chunk the markdown and insert vectorized children.
        chunks = self._chunk_content(content)
        self._insert_chunks(chunks, doc_uuid)
        print(f"Successfully ingested {filename} into {len(chunks)} chunks.")

    def edit_file(self, filename: str, old_string: str, new_string: str, replace_all: bool = False) -> int:
        """Exact string replacement inside a file's raw content, then re-chunks.

        Returns the number of replacements made. Raises if the file is missing,
        the string is not found, or (when replace_all is False) it is ambiguous.
        """
        doc = self.get_document(filename)
        if doc is None:
            raise FileNotFoundError(f"File '{filename}' not found.")

        content = doc.properties["raw_content"]
        count = content.count(old_string)
        if count == 0:
            raise ValueError(f"old_string not found in '{filename}'.")
        if count > 1 and not replace_all:
            raise ValueError(
                f"old_string is not unique in '{filename}' ({count} matches). "
                f"Provide more context or set replace_all=True."
            )

        if replace_all:
            new_content = content.replace(old_string, new_string)
        else:
            new_content = content.replace(old_string, new_string, 1)
            count = 1

        # Update raw content and rebuild the semantic chunks to match.
        self.document_collection.data.update(
            uuid=doc.uuid,
            properties={"raw_content": new_content}
        )
        self._delete_chunks_for(doc.uuid)
        self._insert_chunks(self._chunk_content(new_content), doc.uuid)
        return count

    def _all_documents(self, restrict_to_folder: str = None) -> list:
        """Fetches every stored document (optionally within one folder)."""
        kwargs = {
            "limit": 10000,
            "return_properties": ["filename", "folder_path", "raw_content", "created_at"],
        }
        if restrict_to_folder:
            kwargs["filters"] = Filter.by_property("folder_path").equal(restrict_to_folder)
        return self.document_collection.query.fetch_objects(**kwargs).objects

    def ls(self, folder_path: str = None) -> list:
        """Lists files with metadata (path, size in bytes, created_at)."""
        entries = []
        for obj in self._all_documents(folder_path):
            p = obj.properties
            entries.append({
                "path": f"{p['folder_path']}/{p['filename']}",
                "filename": p["filename"],
                "folder_path": p["folder_path"],
                "size": len(p.get("raw_content") or ""),
                "created_at": p.get("created_at"),
            })
        return sorted(entries, key=lambda e: e["path"])

    def glob(self, pattern: str) -> list:
        """Returns full paths matching a glob pattern (matched on path and filename)."""
        matches = []
        for obj in self._all_documents():
            p = obj.properties
            full_path = f"{p['folder_path']}/{p['filename']}"
            if fnmatch.fnmatch(full_path, pattern) or fnmatch.fnmatch(p["filename"], pattern):
                matches.append(full_path)
        return sorted(matches)

    def grep(self, pattern: str, restrict_to_folder: str = None,
             path_glob: str = None, ignore_case: bool = False) -> list:
        """Regex search across file contents.

        Returns a list of match dicts: {path, line_number, line}. Output
        formatting (content / files / count) is handled by the caller.
        """
        flags = re.IGNORECASE if ignore_case else 0
        regex = re.compile(pattern, flags)
        results = []
        for obj in self._all_documents(restrict_to_folder):
            p = obj.properties
            full_path = f"{p['folder_path']}/{p['filename']}"
            if path_glob and not (
                fnmatch.fnmatch(full_path, path_glob) or fnmatch.fnmatch(p["filename"], path_glob)
            ):
                continue
            for i, line in enumerate((p.get("raw_content") or "").splitlines(), start=1):
                if regex.search(line):
                    results.append({"path": full_path, "line_number": i, "line": line})
        return results

    def read_file_by_name(self, filename: str) -> str:
        """Deterministic exact-match lookup for full file restoration."""
        response = self.document_collection.query.fetch_objects(
            filters=Filter.by_property("filename").equal(filename),
            limit=1
        )
        if not response.objects:
            return "File not found."
        
        return response.objects[0].properties["raw_content"]

    def search_knowledge_base(self, query: str, restrict_to_folder: str = None) -> list:
        """Agentic semantic search across all Markdown chunks."""
        
        # We perform the search on the Chunk collection
        query_obj = self.chunk_collection.query.near_text(
            query=query,
            limit=3,
            return_references=[weaviate.classes.query.QueryReference(link_on="hasDocument")]
        )
        
        results = []
        for obj in query_obj.objects:
            # Extract parent metadata via the cross-reference
            parent_doc = obj.references["hasDocument"].objects[0]
            folder = parent_doc.properties["folder_path"]
            fname = parent_doc.properties["filename"]
            
            if restrict_to_folder and folder != restrict_to_folder:
                continue
                
            results.append({
                "source_file": f"{folder}/{fname}",
                "heading": obj.properties["heading"],
                "content": obj.properties["chunk_text"]
            })
            
        return results

    def list_folders(self) -> list:
        """Returns distinct directory paths for Agent awareness."""
        response = self.document_collection.aggregate.over_all(
            group_by="folder_path"
        )
        return [group.grouped_by.value for group in response.groups]

    def list_files_in_folder(self, folder_path: str) -> list:
        """Returns exact filenames located inside a specific virtual path."""
        response = self.document_collection.query.fetch_objects(
            filters=Filter.by_property("folder_path").equal(folder_path),
            return_properties=["filename"]
        )
        return [obj.properties["filename"] for obj in response.objects]

    def write_chat_message(self, thread_id: str, role: str, content: str, metadata: str = None):
        """Writes a single message to conversational history in Weaviate."""
        self.chat_collection.data.insert(
            properties={
                "thread_id": thread_id,
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or ""
            }
        )

    def read_chat_messages(self, thread_id: str) -> list:
        """Retrieves conversational history for a thread_id, sorted by timestamp."""
        response = self.chat_collection.query.fetch_objects(
            filters=Filter.by_property("thread_id").equal(thread_id),
            limit=100
        )
        # Sort objects by timestamp in Python
        sorted_objs = sorted(
            response.objects,
            key=lambda x: x.properties.get("timestamp", "")
        )
        messages = []
        for obj in sorted_objs:
            messages.append({
                "role": obj.properties["role"],
                "content": obj.properties["content"],
                "timestamp": obj.properties["timestamp"]
            })
        return messages

    def clear_chat_messages(self, thread_id: str):
        """Clears conversational history for a specific thread_id."""
        self.chat_collection.data.delete_many(
            where=Filter.by_property("thread_id").equal(thread_id)
        )


class WeaviateChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, memory_manager: WeaviateMemoryManager, thread_id: str):
        self.memory_manager = memory_manager
        self.thread_id = thread_id

    @property
    def messages(self) -> list[BaseMessage]:
        raw_msgs = self.memory_manager.read_chat_messages(self.thread_id)
        langchain_msgs = []
        for msg in raw_msgs:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                langchain_msgs.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_msgs.append(AIMessage(content=content))
            elif role == "system":
                langchain_msgs.append(SystemMessage(content=content))
            else:
                langchain_msgs.append(HumanMessage(content=content))
        return langchain_msgs

    def add_message(self, message: BaseMessage) -> None:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, SystemMessage):
            role = "system"
        else:
            role = "user"
        self.memory_manager.write_chat_message(self.thread_id, role, message.content)

    def clear(self) -> None:
        self.memory_manager.clear_chat_messages(self.thread_id)
