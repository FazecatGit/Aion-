import re
import json
from langchain_ollama import OllamaLLM

from brain.config import LLM_MODEL, DATA_DIR
from brain.pdf_utils import load_pdfs
from brain.fast_search import fast_topic_search
from db.db_reader import get_code_documents
from .tools import read_file, write_file, run_git_command, show_diff, run_python_file, run_shell_command, list_files

class CodeAgent:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    def edit_code(self, path: str, instruction: str, dry_run: bool = True, use_rag: bool = False):
        source = read_file(path)

        rag_context = ""
        if use_rag:
            print("\nSearching offline PDFs for context...")
            try:
                documents = load_pdfs(DATA_DIR)
                
                results = fast_topic_search(instruction, documents)
                
                if results:
                    rag_context = "REFERENCE DOCUMENTATION (Use this to guide your code edit):\n"
                    for i, doc in enumerate(results):
                        rag_context += f"--- Source {i+1} ---\n{doc['content']}\n\n"
                    print(f"Found {len(results)} relevant PDF chunks!")
                else:
                    print("No relevant PDF context found. Proceeding without RAG.")
            except Exception as e:
                print(f"Failed to load RAG context: {e}. Proceeding without RAG.")

        prompt = (
            "You are an expert AI coding agent. I need you to edit a file.\n"
            "You must output a SEARCH/REPLACE block that specifies exactly what code to remove, and what code to insert.\n\n"
            "1. The SEARCH block must EXACTLY match the existing code in the file, character for character, including indentation.\n"
            "2. Include enough context (a few lines before and after) so the SEARCH block is unique in the file.\n"
            "3. If you are ADDING new code, you MUST include the original surrounding code inside the REPLACE block so it doesn't get deleted.\n"
            "4. INTERPRET INSTRUCTIONS LITERALLY: If asked to add a specific print statement, string, or command, type it EXACTLY as requested. Do not invent function names, do not turn it into an import, and do not try to be clever.\n"
            "5. Output ONLY the SEARCH/REPLACE block inside a single markdown block. No conversational text.\n\n"
            "FORMAT:\n"
            "```python\n"
            "<<<<<<< SEARCH\n"
            "(exact lines of old code here)\n"
            "=======\n"
            "(new edited code here)\n"
            ">>>>>>> REPLACE\n"
            "```\n\n"
            f"INSTRUCTION: {instruction}\n\n"
            f"{rag_context}"
            "ORIGINAL FILE CONTENT:\n"
            f"```python\n{source}\n```\n"
        )
        
        llm = OllamaLLM(model=LLM_MODEL, temperature=0.0)
        print("Waiting for LLM to calculate the exact edit...")
        raw_output = llm.invoke(prompt)

        # Parse the output
        search_match = re.search(r'<<<<<<< SEARCH\n(.*?)\n=======', raw_output, re.DOTALL)
        replace_match = re.search(r'=======\n(.*?)\n>>>>>>> REPLACE', raw_output, re.DOTALL)

        if not search_match or not replace_match:
            print("\nERROR: LLM failed to format the Search/Replace block correctly.")
            print("Raw LLM Output was:\n", raw_output)
            return source

        search_text = search_match.group(1)
        replace_text = replace_match.group(1)

        # Apply the edit to the source code
        match_start_idx = -1
        source_lines = source.split('\n')
        search_lines = search_text.strip().split('\n')
        replace_lines = replace_text.strip("\n").split('\n')
        
        for i in range(len(source_lines) - len(search_lines) + 1):
            match = True
            for j in range(len(search_lines)):
                if source_lines[i+j].strip() != search_lines[j].strip():
                    match = False
                    break
            if match:
                match_start_idx = i
                break

        if match_start_idx == -1:
            print("\nERROR: The SEARCH block provided by the LLM does not match any code in the file.")
            print("It tried to find this text (ignoring indentation):\n", search_text)
            return source

        original_indent = len(source_lines[match_start_idx]) - len(source_lines[match_start_idx].lstrip())
        indent_str = " " * original_indent

        indented_replace = []
        for line in replace_lines:
            if line.strip() == "":
                indented_replace.append("")
            else:
                llm_indent = len(line) - len(line.lstrip())
                indented_replace.append(indent_str + (" " * llm_indent) + line.lstrip())

        new_source_lines = (
            source_lines[:match_start_idx] + 
            indented_replace + 
            source_lines[match_start_idx + len(search_lines):]
        )
        
        new_source = "\n".join(new_source_lines)

        diff = show_diff(source, new_source)
        print("\n--- DIFF PREVIEW ---")
        print(diff or "No changes detected.")
        print("--------------------\n")

        if dry_run:
            print("Dry run mode. No file written.")
            return new_source


        write_file(path, new_source)
        run_git_command(["git", "diff", path])
        return new_source

    def list_db_code(self, limit: int = 20):
        docs = get_code_documents(limit=limit)
        return [d["text"] for d in docs]
        
    def test_file(self, path: str) -> str:
        return run_python_file(path)

    def run_shell_command(self, command: str) -> str:
        return run_shell_command(command)
    
    def list_files(self) -> list[str]:
        return list_files(self.repo_path)
