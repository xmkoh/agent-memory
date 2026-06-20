import os
import sys
import argparse
import asyncio
import weaviate
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from memory_manager import WeaviateMemoryManager, WeaviateChatMessageHistory

# Ensure API credentials are set
token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
if not token:
    print("Error: ANTHROPIC_AUTH_TOKEN environment variable not set.")
    sys.exit(1)

# Configure the LLM using Zenmux API
llm = ChatOpenAI(
    model="deepseek/deepseek-v4-pro",
    api_key=token,
    base_url="https://zenmux.ai/api/v1",
    temperature=0.2
)

# Connect to Weaviate
client = weaviate.connect_to_local()
memory_manager = WeaviateMemoryManager(client)

# Define LangChain Memory Tools
@tool
def write_memory_file(filename: str, folder_path: str, content: str) -> str:
    """
    Saves a document or file content to the long-term Weaviate memory store.
    It saves the raw text and automatically creates semantic vector chunks.
    If a file with the same name exists it is overwritten (upsert), so repeated
    saves do not accumulate duplicates.
    Use this to persist knowledge, architecture documents, notes, or findings.
    """
    try:
        memory_manager.write_markdown(filename, folder_path, content, overwrite=True)
        return f"Successfully saved file '{filename}' to folder '{folder_path}' in long-term memory."
    except Exception as e:
        return f"Error writing file to memory: {e}"

@tool
def search_memory(query: str, restrict_to_folder: str = None) -> str:
    """
    Performs a semantic search over all documents in long-term memory.
    You can optionally restrict the search to a specific folder path.
    """
    try:
        results = memory_manager.search_knowledge_base(query, restrict_to_folder)
        if not results:
            return "No matching records found in long-term memory."
        
        output = []
        for r in results:
            output.append(f"Source: {r['source_file']} | Section: {r['heading']}\nContent: {r['content']}\n---")
        return "\n".join(output)
    except Exception as e:
        return f"Error searching memory: {e}"

@tool
def list_memory_folders() -> str:
    """
    Lists all directory/folder paths stored in the long-term memory database.
    """
    try:
        folders = memory_manager.list_folders()
        if not folders:
            return "No folders found in memory."
        return f"Folders in memory: {folders}"
    except Exception as e:
        return f"Error listing folders: {e}"

# --- Deepagents-style filesystem tools (backed by the Weaviate document store) ---

@tool
def ls(folder_path: str = None) -> str:
    """
    Lists files in the memory store with metadata (path, size in bytes, created_at).
    Optionally restrict to a single folder_path. Use this to see what files exist.
    """
    try:
        entries = memory_manager.ls(folder_path)
        if not entries:
            return "No files found."
        lines = [f"{e['path']}\t{e['size']} bytes\t{e['created_at']}" for e in entries]
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing files: {e}"

@tool
def read_file(filename: str, offset: int = 0, limit: int = 2000) -> str:
    """
    Reads a file's contents with 1-based line numbers (cat -n style).
    For large files, use offset (0-based line index to start from) and limit
    (max number of lines to return) to page through the content.
    """
    try:
        content = memory_manager.read_file_by_name(filename)
        if content == "File not found.":
            return f"Error: file '{filename}' not found."
        all_lines = content.splitlines()
        window = all_lines[offset:offset + limit]
        if not window:
            return f"(no lines in range: offset={offset}, limit={limit}; file has {len(all_lines)} lines)"
        numbered = [f"{offset + i + 1:6d}\t{line}" for i, line in enumerate(window)]
        out = "\n".join(numbered)
        if offset + limit < len(all_lines):
            out += f"\n... ({len(all_lines) - (offset + limit)} more lines; increase offset/limit to continue)"
        return out
    except Exception as e:
        return f"Error reading file: {e}"

@tool
def write_file(filename: str, folder_path: str, content: str) -> str:
    """
    Creates a NEW file in the memory store. Fails if a file with the same name
    already exists (use edit_file to modify an existing file instead).
    If the file was previously read this session, the write is automatically
    rejected with a ConflictError if the stored content changed since that read.
    """
    try:
        memory_manager.write_markdown(filename, folder_path, content, overwrite=False)
        return f"Successfully created '{filename}' in '{folder_path}'."
    except FileExistsError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error writing file: {e}"

@tool
def edit_file(filename: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """
    Performs an exact string replacement inside an existing file.
    By default old_string must appear exactly once (otherwise the edit fails as
    ambiguous); set replace_all=True to replace every occurrence. The file's
    semantic chunks are automatically re-indexed after the edit.
    If the file was previously read this session, the edit is automatically
    rejected with a ConflictError if the stored content changed since that read.
    """
    try:
        n = memory_manager.edit_file(filename, old_string, new_string, replace_all)
        return f"Made {n} replacement(s) in '{filename}'."
    except Exception as e:
        return f"Error editing file: {e}"

@tool
def glob(pattern: str) -> str:
    """
    Finds files whose path or name matches a glob pattern (e.g. '*.md',
    'projects/alpha/*'). Returns matching full paths.
    """
    try:
        matches = memory_manager.glob(pattern)
        if not matches:
            return f"No files match pattern '{pattern}'."
        return "\n".join(matches)
    except Exception as e:
        return f"Error running glob: {e}"

@tool
def grep(pattern: str, output_mode: str = "content", restrict_to_folder: str = None,
         path_glob: str = None, ignore_case: bool = False) -> str:
    """
    Searches file contents with a regular expression.
    output_mode controls the result format:
      - "content" (default): each match as 'path:line_number: line'
      - "files_with_matches": just the unique file paths that contain a match
      - "count": 'path: count' per file with matches
    Optionally restrict_to_folder, filter files by path_glob, or set ignore_case=True.
    """
    try:
        results = memory_manager.grep(pattern, restrict_to_folder, path_glob, ignore_case)
        if not results:
            return f"No matches for '{pattern}'."

        if output_mode == "files_with_matches":
            seen = sorted({r["path"] for r in results})
            return "\n".join(seen)
        if output_mode == "count":
            counts = {}
            for r in results:
                counts[r["path"]] = counts.get(r["path"], 0) + 1
            return "\n".join(f"{path}: {n}" for path, n in sorted(counts.items()))
        # default: content mode
        return "\n".join(f"{r['path']}:{r['line_number']}: {r['line']}" for r in results)
    except Exception as e:
        return f"Error running grep: {e}"

tools = [
    write_memory_file,
    search_memory,
    list_memory_folders,
    ls,
    read_file,
    write_file,
    edit_file,
    glob,
    grep,
]

# Map tool names to objects for execution
tool_map = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)


# Conversational Agent Loop with Memory persistence
def execute_agent(user_input: str, thread_id: str) -> str:
    # 1. Fetch short-term conversational history from Weaviate
    history = WeaviateChatMessageHistory(memory_manager, thread_id)
    chat_history = history.messages
    
    # 2. Reconstruct messages list for current execution (system prompt + history + current input)
    system_prompt = (
        "You are an advanced AI research assistant with access to two types of memory:\n"
        "1. Short-term conversational memory: Automatically tracks the context of this conversation thread.\n"
        "2. Long-term file memory: A Weaviate-backed document store containing files organized in directories.\n"
        "You can search, list, read, and write files to your long-term memory using the provided tools.\n"
        "Always check your memory or read relevant files if you need context to answer the user's questions."
    )
    
    messages = list(chat_history)
    messages.append(HumanMessage(content=user_input))
    
    # 3. Execution loop (supports up to 5 steps of tool usage per turn)
    max_iterations = 5
    for _ in range(max_iterations):
        all_messages = [SystemMessage(content=system_prompt)] + messages
        response = llm_with_tools.invoke(all_messages)
        
        # If no tools to call, this is the final agent response
        if not response.tool_calls:
            # Persist this turn in conversational memory
            history.add_message(HumanMessage(content=user_input))
            history.add_message(AIMessage(content=response.content))
            return response.content
        
        # Append AI tool-call message
        messages.append(response)
        
        # Execute each tool call and append the ToolMessage
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            
            print(f" -> Agent calling tool '{tool_name}' with args {tool_args}...")
            if tool_name in tool_map:
                try:
                    tool_output = tool_map[tool_name].invoke(tool_args)
                except Exception as e:
                    tool_output = f"Error executing tool: {e}"
            else:
                tool_output = f"Error: Tool '{tool_name}' not found."
                
            messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_id))
            
    fallback_msg = "Agent reached maximum tool call iterations without resolving."
    history.add_message(HumanMessage(content=user_input))
    history.add_message(AIMessage(content=fallback_msg))
    return fallback_msg

def run_interactive():
    print("====================================================")
    print("Welcome to LangChain Deep Agent (Weaviate Memory)")
    print("Type 'exit' to quit. Session ID is 'default_session'.")
    print("====================================================")
    
    session_id = "default_session"
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.strip().lower() == "exit":
                break
            if not user_input.strip():
                continue
                
            output = execute_agent(user_input, session_id)
            print(f"\nAgent: {output}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {e}")
            
    client.close()

def run_tests():
    print("--- Starting Integration Tests ---")
    session_id = "test_session_99"
    
    # Clean up previous test thread
    hist = WeaviateChatMessageHistory(memory_manager, session_id)
    hist.clear()
    
    # 1. Ingest a project specification file to long-term memory
    print("\nTest 1: Writing a file to long-term Weaviate memory...")
    write_res = write_memory_file.invoke({
        "filename": "team_alpha_spec.md",
        "folder_path": "projects/alpha",
        "content": """# Project Alpha Specifications
## Core Engine
We are developing the core orchestration engine using LangChain and Weaviate.
## Deployment
The production deployment will run on AWS ECS with auto-scaling groups.
"""
    })
    print(write_res)
    assert "Successfully saved" in write_res, f"write_memory_file did not succeed: {write_res}"

    # 2. Check directory mapping
    print("\nTest 2: Listing folders in memory...")
    folders_res = list_memory_folders.invoke({})
    print(folders_res)
    assert "projects/alpha" in folders_res, "Expected folder projects/alpha not found"
    
    # 3. Check files in folder
    print("\nTest 3: Listing files in folder...")
    files_res = ls.invoke({"folder_path": "projects/alpha"})
    print(files_res)
    assert "team_alpha_spec.md" in files_res, "Expected file team_alpha_spec.md not found"
    
    # 4. Agent semantic search query
    print("\nTest 4: Agent semantic search on memory...")
    search_res = search_memory.invoke({"query": "What engine are we developing?"})
    print(search_res)
    assert "core orchestration engine" in search_res or "LangChain" in search_res, "Expected context not found in search results"
    
    # 5. Agent Chat Session - Turn 1 (Introduce user preference)
    print("\nTest 5: Conversational Memory - Turn 1...")
    res1 = execute_agent(
        "Hi, I am lead engineer Bob. Remember that I prefer AWS deployments.",
        session_id
    )
    print(f"Agent response: {res1}")
    
    # 6. Agent Chat Session - Turn 2 (Check if agent remembers Bob and prefers AWS, and looks up spec)
    print("\nTest 6: Conversational Memory - Turn 2...")
    res2 = execute_agent(
        "Can you check our Project Alpha spec and tell me where we are deploying it? Also, who am I?",
        session_id
    )
    print(f"Agent response: {res2}")
    
    # Validations on output
    output_lower = res2.lower()
    assert "bob" in output_lower, "Agent failed to remember the user's name (Bob)"
    assert "aws" in output_lower or "ecs" in output_lower, "Agent failed to find AWS/ECS deployment from memory spec"

    # --- Filesystem tool tests (deterministic, no LLM involved) ---
    fs_folder = "fs/tests"
    fs_files = ["fs_notes.md", "fs_log.txt"]

    def _cleanup_fs_files():
        for fn in fs_files:
            doc = memory_manager.get_document(fn)
            if doc is not None:
                memory_manager._delete_chunks_for(doc.uuid)
                memory_manager.document_collection.data.delete_by_id(doc.uuid)

    _cleanup_fs_files()  # ensure a clean slate from prior runs

    # 7. write_file creates a new file
    print("\nTest 7: write_file creates a new file...")
    w_res = write_file.invoke({
        "filename": "fs_notes.md",
        "folder_path": fs_folder,
        "content": "# Notes\nalpha line\n## Section\nbeta line\nbeta again\n"
    })
    print(w_res)
    assert "Successfully created" in w_res, "write_file did not confirm creation"

    # 8. write_file rejects a duplicate filename
    print("\nTest 8: write_file rejects duplicate...")
    dup_res = write_file.invoke({
        "filename": "fs_notes.md", "folder_path": fs_folder, "content": "dup"
    })
    print(dup_res)
    assert "already exists" in dup_res, "Duplicate write_file was not rejected"

    # Second file for glob/grep coverage across files
    write_file.invoke({
        "filename": "fs_log.txt", "folder_path": fs_folder,
        "content": "line one\nbeta in log\nline three\n"
    })

    # 9. ls returns the file with metadata
    print("\nTest 9: ls returns files with metadata...")
    ls_res = ls.invoke({"folder_path": fs_folder})
    print(ls_res)
    assert "fs/tests/fs_notes.md" in ls_res, "ls missing fs_notes.md"
    assert "bytes" in ls_res, "ls missing size metadata"

    # 10. glob finds matching files
    print("\nTest 10: glob '*.md'...")
    glob_res = glob.invoke({"pattern": "*.md"})
    print(glob_res)
    assert "fs/tests/fs_notes.md" in glob_res, "glob did not match fs_notes.md"
    assert "fs/tests/fs_log.txt" not in glob_res, "glob '*.md' wrongly matched a .txt file"

    # 11. grep across all three output modes
    print("\nTest 11: grep output modes...")
    content_mode = grep.invoke({"pattern": "beta", "restrict_to_folder": fs_folder})
    print("content:\n" + content_mode)
    assert "fs/tests/fs_notes.md:" in content_mode, "grep content mode missing path:line"

    files_mode = grep.invoke({
        "pattern": "beta", "output_mode": "files_with_matches", "restrict_to_folder": fs_folder
    })
    print("files_with_matches:\n" + files_mode)
    assert "fs/tests/fs_notes.md" in files_mode and "fs/tests/fs_log.txt" in files_mode, \
        "grep files_with_matches missing a file"

    count_mode = grep.invoke({
        "pattern": "beta", "output_mode": "count", "restrict_to_folder": fs_folder
    })
    print("count:\n" + count_mode)
    assert "fs/tests/fs_notes.md: 2" in count_mode, "grep count mode wrong (expected 2 in fs_notes.md)"

    # 12. read_file with offset/limit returns a numbered window
    print("\nTest 12: read_file offset/limit numbering...")
    read_res = read_file.invoke({"filename": "fs_notes.md", "offset": 1, "limit": 2})
    print(read_res)
    assert "     2\talpha line" in read_res, "read_file did not number line 2 from the offset"
    assert "# Notes" not in read_res, "read_file offset=1 did not skip line 1"

    # 13. edit_file: rejects non-unique, succeeds on unique, replace_all replaces all
    print("\nTest 13: edit_file unique / ambiguous / replace_all...")
    ambig = edit_file.invoke({
        "filename": "fs_notes.md", "old_string": "beta", "new_string": "BETA"
    })
    print(ambig)
    assert "not unique" in ambig, "edit_file did not reject a non-unique old_string"

    unique = edit_file.invoke({
        "filename": "fs_notes.md", "old_string": "alpha line", "new_string": "gamma line"
    })
    print(unique)
    assert "Made 1 replacement" in unique, "edit_file unique replace failed"

    all_res = edit_file.invoke({
        "filename": "fs_notes.md", "old_string": "beta", "new_string": "BETA", "replace_all": True
    })
    print(all_res)
    assert "Made 2 replacement" in all_res, "edit_file replace_all did not replace both occurrences"

    # 14. semantic search sees the edited content (re-chunking worked)
    print("\nTest 14: search reflects edited content...")
    edited_search = search_memory.invoke({"query": "gamma line"})
    print(edited_search)
    assert "gamma" in edited_search.lower(), "Edit was not re-indexed for semantic search"

    _cleanup_fs_files()
    print("Cleaned up filesystem test files.")

    # --- Conflict-check tests ---
    conflict_file = "conflict_check.md"
    conflict_folder = "fs/tests"

    def _cleanup_conflict_file():
        doc = memory_manager.get_document(conflict_file)
        if doc is not None:
            memory_manager._delete_chunks_for(doc.uuid)
            memory_manager.document_collection.data.delete_by_id(doc.uuid)

    def _external_modify(new_content: str):
        """Simulates an out-of-band edit by updating Weaviate directly,
        bypassing the manager so _read_hashes is not refreshed."""
        doc = memory_manager.get_document(conflict_file)
        memory_manager.document_collection.data.update(
            uuid=doc.uuid,
            properties={"raw_content": new_content}
        )

    _cleanup_conflict_file()

    # 15. ConflictError fires on edit when file changed after read
    print("\nTest 15: ConflictError on edit after external modification...")
    write_file.invoke({
        "filename": conflict_file, "folder_path": conflict_folder,
        "content": "# Conflict\noriginal content\n"
    })
    read_file.invoke({"filename": conflict_file})          # records hash
    _external_modify("# Conflict\nmodified externally\n")  # out-of-band change
    conflict_edit_res = edit_file.invoke({
        "filename": conflict_file, "old_string": "original content", "new_string": "new content"
    })
    print(conflict_edit_res)
    assert "Conflict" in conflict_edit_res, "edit_file did not raise ConflictError after external modification"

    # 16. ConflictError fires on overwrite-write when file changed after read
    print("\nTest 16: ConflictError on write (overwrite) after external modification...")
    # _read_hashes still holds the old hash from test 15 read
    conflict_write_res = write_memory_file.invoke({
        "filename": conflict_file, "folder_path": conflict_folder,
        "content": "# Conflict\noverwrite attempt\n"
    })
    print(conflict_write_res)
    assert "Conflict" in conflict_write_res, "write_memory_file did not raise ConflictError after external modification"

    # 17. Re-reading the file clears the conflict — edit succeeds afterwards
    print("\nTest 17: Re-reading clears conflict, subsequent edit succeeds...")
    read_file.invoke({"filename": conflict_file})  # re-read records new hash
    reread_edit_res = edit_file.invoke({
        "filename": conflict_file, "old_string": "modified externally", "new_string": "restored content"
    })
    print(reread_edit_res)
    assert "Made 1 replacement" in reread_edit_res, "edit_file failed after re-read cleared the conflict"

    # 18. No conflict check fires when the file has never been read this session
    print("\nTest 18: No conflict check when file was not read this session...")
    # Simulate a file that exists in the store but has no entry in _read_hashes
    # by inserting it directly and then removing it from the cache.
    direct_file = "direct_insert.md"
    direct_doc = memory_manager.get_document(direct_file)
    if direct_doc is not None:
        memory_manager._delete_chunks_for(direct_doc.uuid)
        memory_manager.document_collection.data.delete_by_id(direct_doc.uuid)
    write_file.invoke({
        "filename": direct_file, "folder_path": conflict_folder,
        "content": "# Direct\nsome content\n"
    })
    memory_manager._read_hashes.pop(direct_file, None)  # clear the cached hash
    no_conflict_res = edit_file.invoke({
        "filename": direct_file, "old_string": "some content", "new_string": "edited content"
    })
    print(no_conflict_res)
    assert "Made 1 replacement" in no_conflict_res, "edit_file incorrectly blocked an edit with no prior read"

    # clean up
    _cleanup_conflict_file()
    for fn in [direct_file]:
        doc = memory_manager.get_document(fn)
        if doc is not None:
            memory_manager._delete_chunks_for(doc.uuid)
            memory_manager.document_collection.data.delete_by_id(doc.uuid)
    print("Cleaned up conflict-check test files.")

    # --- WeaviateStore tests ---
    # WeaviateStore wraps memory_manager — no separate collection, no mock embedder.
    # Semantic search uses the existing Chunk collection (text2vec-transformers).
    from weaviate_store import WeaviateStore
    from weaviate.classes.query import Filter as _Filter

    store = WeaviateStore(memory_manager)

    def _cleanup_store(*namespaces):
        for ns in namespaces:
            folder = "/".join(ns)
            docs = memory_manager.document_collection.query.fetch_objects(
                filters=_Filter.by_property("folder_path").equal(folder),
                limit=1000,
            ).objects
            for obj in docs:
                memory_manager._delete_chunks_for(obj.uuid)
                memory_manager.document_collection.data.delete_by_id(obj.uuid)

    ns_a = ("store_test", "alpha")
    ns_b = ("store_test", "beta")
    ns_deep = ("store_test", "alpha", "deep")
    _cleanup_store(ns_a, ns_b, ns_deep)

    # 19. put and get round-trip
    print("\nTest 19: WeaviateStore put and get...")
    store.put(ns_a, "k1", {"msg": "hello world"}, index=True)
    item = store.get(ns_a, "k1")
    assert item is not None, "get returned None after put"
    assert item.key == "k1", "item key mismatch"
    assert item.namespace == ns_a, "item namespace mismatch"
    assert item.value == {"msg": "hello world"}, "item value mismatch"
    assert item.created_at is not None, "created_at missing"
    print(f"  got: {item.value}")

    # 20. get on missing key returns None
    print("\nTest 20: WeaviateStore get on missing key returns None...")
    missing = store.get(ns_a, "does_not_exist")
    assert missing is None, "get on missing key did not return None"
    print("  returned None as expected")

    # 21. put upserts (second put with same key updates value)
    print("\nTest 21: WeaviateStore upsert via repeated put...")
    store.put(ns_a, "k1", {"msg": "updated"}, index=True)
    updated = store.get(ns_a, "k1")
    assert updated.value == {"msg": "updated"}, "upsert did not update value"
    print(f"  updated value: {updated.value}")

    # 22. delete removes the item
    print("\nTest 22: WeaviateStore delete...")
    store.put(ns_a, "to_delete", {"x": 1})
    store.delete(ns_a, "to_delete")
    assert store.get(ns_a, "to_delete") is None, "item still present after delete"
    store.delete(ns_a, "to_delete")  # deleting a non-existent key must not raise
    print("  deleted successfully; double-delete did not raise")

    # 23. list_namespaces returns distinct namespaces
    print("\nTest 23: WeaviateStore list_namespaces...")
    store.put(ns_a, "x", {"v": 1})
    store.put(ns_b, "y", {"v": 2})
    store.put(ns_deep, "z", {"v": 3})
    all_ns = store.list_namespaces()
    assert ns_a in all_ns, f"ns_a missing from list_namespaces: {all_ns}"
    assert ns_b in all_ns, f"ns_b missing from list_namespaces: {all_ns}"
    print(f"  namespaces: {all_ns}")

    # 24. list_namespaces prefix filter
    print("\nTest 24: WeaviateStore list_namespaces with prefix...")
    prefix_ns = store.list_namespaces(prefix=("store_test",))
    assert ns_a in prefix_ns, "ns_a missing with prefix filter"
    assert ns_b in prefix_ns, "ns_b missing with prefix filter"
    assert all(ns[0] == "store_test" for ns in prefix_ns), \
        "prefix filter let through wrong namespace"
    print(f"  filtered: {prefix_ns}")

    # 25. list_namespaces max_depth truncates namespace tuples
    print("\nTest 25: WeaviateStore list_namespaces max_depth=1...")
    depth1 = store.list_namespaces(prefix=("store_test",), max_depth=1)
    assert all(len(ns) == 1 for ns in depth1), \
        f"max_depth=1 did not truncate: {depth1}"
    assert ("store_test",) in depth1, "truncated root namespace missing"
    print(f"  depth-1 namespaces: {depth1}")

    # 26. search without query returns items by namespace prefix (no ranking)
    print("\nTest 26: WeaviateStore search without query...")
    unranked = store.search(("store_test",))
    paths = [(r.namespace, r.key) for r in unranked]
    assert any(ns == ns_a for ns, _ in paths), "ns_a items missing from unranked search"
    assert any(ns == ns_b for ns, _ in paths), "ns_b items missing from unranked search"
    print(f"  returned {len(unranked)} items")

    # 27. search with query returns ranked SearchItems with scores
    print("\nTest 27: WeaviateStore search with query (near_text via text2vec-transformers)...")
    store.put(ns_a, "doc_cat",  {"content": "cats are fluffy animals"}, index=["content"])
    store.put(ns_a, "doc_dog",  {"content": "dogs are loyal companions"}, index=["content"])
    store.put(ns_a, "doc_car",  {"content": "cars have four wheels"}, index=["content"])
    ranked = store.search(("store_test", "alpha"), query="feline pets")
    assert len(ranked) > 0, "near_vector search returned no results"
    assert all(r.score is not None for r in ranked), "SearchItem scores missing"
    # Scores should be in descending order (Weaviate returns highest certainty first)
    scores = [r.score for r in ranked]
    assert scores == sorted(scores, reverse=True), f"results not sorted by score: {scores}"
    print(f"  top result: {ranked[0].value!r} (score={ranked[0].score:.4f})")

    # 28. items stored with index=False have no vector and are excluded from near_vector
    print("\nTest 28: index=False items excluded from vector search...")
    store.put(ns_a, "no_vec", {"content": "feline cats pets kittens"}, index=False)
    ranked2 = store.search(("store_test", "alpha"), query="feline pets")
    no_vec_keys = [r.key for r in ranked2]
    assert "no_vec" not in no_vec_keys, \
        "index=False item incorrectly appeared in near_vector results"
    print("  index=False item correctly excluded from ranked search")

    # 29. Async contract: aput / aget / asearch / adelete round-trip
    # These are the methods LangGraph actually calls under async graph execution
    # (deepagents.create_deep_agent runs the graph with ainvoke).
    print("\nTest 29: WeaviateStore async methods (aput/aget/asearch/adelete)...")
    ns_async = ("store_test", "async")

    async def _async_checks():
        await store.aput(ns_async, "ak1", {"content": "async fluffy cats"}, index=["content"])
        got = await store.aget(ns_async, "ak1")
        assert got is not None and got.value == {"content": "async fluffy cats"}, \
            "aget did not round-trip aput"
        found = await store.asearch(ns_async, query="feline kittens")
        assert any(r.key == "ak1" for r in found), "asearch did not find the async item"
        ns_list = await store.alist_namespaces(prefix=("store_test",))
        assert ns_async in ns_list, "alist_namespaces missing the async namespace"
        await store.adelete(ns_async, "ak1")
        assert await store.aget(ns_async, "ak1") is None, "adelete did not remove the item"

    asyncio.run(_async_checks())
    print("  async aput/aget/asearch/alist_namespaces/adelete all passed")

    # 30. batch / abatch dispatch over Op types (the abstract BaseStore surface)
    print("\nTest 30: WeaviateStore batch/abatch dispatch...")
    from langgraph.store.base import GetOp, PutOp, SearchOp, ListNamespacesOp

    ns_batch = ("store_test", "batch")
    _cleanup_store(ns_batch)

    # batch: a PutOp followed by a GetOp for the same key
    put_res, get_res = store.batch([
        PutOp(ns_batch, "bk1", {"content": "batched value"}, ["content"]),
        GetOp(ns_batch, "bk1"),
    ])
    assert put_res is None, "PutOp should return None"
    assert get_res is not None and get_res.value == {"content": "batched value"}, \
        "GetOp via batch did not return the put value"

    # batch: SearchOp and ListNamespacesOp
    search_res = store.batch([SearchOp(ns_batch, query="batched")])[0]
    assert any(r.key == "bk1" for r in search_res), "SearchOp via batch found nothing"

    # PutOp with value=None is the delete convention
    store.batch([PutOp(ns_batch, "bk1", None)])
    assert store.get(ns_batch, "bk1") is None, "PutOp(value=None) did not delete"

    # abatch mirrors batch on the async path
    async def _abatch_checks():
        res = await store.abatch([
            PutOp(ns_batch, "bk2", {"content": "abatched"}, ["content"]),
            GetOp(ns_batch, "bk2"),
        ])
        assert res[1] is not None and res[1].value == {"content": "abatched"}, \
            "abatch GetOp did not round-trip"

    asyncio.run(_abatch_checks())
    print("  batch/abatch dispatch over Get/Put/Search/ListNamespaces passed")

    _cleanup_store(ns_a, ns_b, ns_deep, ns_async, ns_batch)
    print("Cleaned up WeaviateStore test items.")

    print("\nAll integration tests passed successfully!")
    client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run automated integration tests")
    args = parser.parse_args()
    
    if args.test:
        run_tests()
    else:
        run_interactive()
