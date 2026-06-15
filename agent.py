import os
import sys
import argparse
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
    Use this to persist knowledge, architecture documents, notes, or findings.
    """
    try:
        memory_manager.write_markdown(filename, folder_path, content)
        return f"Successfully saved file '{filename}' to folder '{folder_path}' in long-term memory."
    except Exception as e:
        return f"Error writing file to memory: {e}"

@tool
def read_memory_file(filename: str) -> str:
    """
    Retrieves the full raw content of a specific file from long-term memory by its name.
    """
    try:
        content = memory_manager.read_file_by_name(filename)
        return content
    except Exception as e:
        return f"Error reading file from memory: {e}"

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

@tool
def list_memory_files(folder_path: str) -> str:
    """
    Lists all filenames stored in a specific folder inside the long-term memory database.
    """
    try:
        files = memory_manager.list_files_in_folder(folder_path)
        if not files:
            return f"No files found in folder '{folder_path}'."
        return f"Files in '{folder_path}': {files}"
    except Exception as e:
        return f"Error listing files: {e}"

tools = [
    write_memory_file,
    read_memory_file,
    search_memory,
    list_memory_folders,
    list_memory_files
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
    
    # 2. Check directory mapping
    print("\nTest 2: Listing folders in memory...")
    folders_res = list_memory_folders.invoke({})
    print(folders_res)
    assert "projects/alpha" in folders_res, "Expected folder projects/alpha not found"
    
    # 3. Check files in folder
    print("\nTest 3: Listing files in folder...")
    files_res = list_memory_files.invoke({"folder_path": "projects/alpha"})
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
