import asyncio
import warnings

from brain.ingest import ingest_docs
from brain.config import DATA_DIR, LLM_MODEL
from agent.code_agent import CodeAgent
from langchain_ollama import OllamaLLM
from brain.augmented_generation_query import query_brain_comprehensive, _last_filters, session_chat_history
from brain.pdf_utils import load_pdfs

warnings.filterwarnings("ignore", message=".*Odd-length string.*")
warnings.filterwarnings("ignore", message=".*invalid hex string.*")

async def main():
    raw_docs = None
    while True:
        print("Aion RAG System")
        print("=" * 50)
        print("\n1: Ingest Documents")
        print("2: Query (Comprehensive)")
        print("3: Code Agent")
        print("4: Quit Application") 
        print("\nEnter choice: ", end="")

        choice = input().strip()
        if choice == "1":
            # topic_index not neeeded yet - just load topic synonyms as part of ingest_docs
            documents, topic_synonyms = ingest_docs()
            raw_docs = load_pdfs(DATA_DIR)
            print("\nIngestion complete. Topics loaded:")
            print("Available topics:", list(topic_synonyms.keys()))
            print("Returning to main menu...")

        elif choice == "2":
            print("\nQuery mode. Type 'quit' to return to main menu.\nType 'verbose' to toggle debug info.\n")
            verbose = False

            while True:
                q = input("Ask: ").strip()
                if q.lower() == "quit":
                    session_chat_history.clear()
                    print("Returning to main menu...")
                    break
                
                if q.lower() == "verbose":
                    verbose = not verbose
                    print(f"Verbose mode: {verbose}\n")
                    continue

                if q:
                    results = await query_brain_comprehensive(q, verbose=verbose, raw_docs=raw_docs, session_chat_history=session_chat_history)
                    print("\n" + "=" * 50)
                    print("\n DIRECT ANSWER\n")
                    print(results["answer"])
                    print("\n SUMMARY\n")
                    print(results["summary"])
                    print("\n KEY CITATIONS\n")
                    print(results["citations"])
                    print("\n DETAILED EXPLANATION\n")
                    print(results["detailed"])
                    print("\n" + "=" * 50 + "\n")

        elif choice == "3":
            print("\nCode Agent mode. Type 'quit' to return to main menu.\n")
            
            repo_path = input("Enter the path to your repository (e.g., .): ").strip()
            if repo_path.lower() == "quit":
                continue
                
            try:
                agent = CodeAgent(repo_path)
            except Exception as e:
                print(f"Could not initialize agent: {e}")
                continue

            while True:

                filepath = input("\nEnter specific file to edit (e.g., main.py) or 'quit': ").strip()
                if filepath.lower() == "quit":
                    print("Exiting Code Agent mode. Returning to main menu...")
                    break
                if not filepath:
                    continue

                instruction = input("Enter your instruction for the code edit: ").strip()
                if not instruction:
                    continue
                    
                use_rag_input = input("Search your PDFs for context to help write this code? (y/n): ").strip().lower()
                use_rag = True if use_rag_input == 'y' else False
                    
                print("\nProcessing your request...\n")

                try:
                    agent.edit_code(path=filepath, instruction=instruction, dry_run=True, use_rag=use_rag, session_chat_history=session_chat_history)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    continue


                confirm = input("\nDo you want to apply these changes? (y/n): ").strip().lower()
                if confirm == "y":
                    try:
                        agent.edit_code(path=filepath, instruction=instruction, dry_run=False, use_rag=use_rag, session_chat_history=session_chat_history)
                        print("Edit applied successfully.")
                    except Exception as e:
                        print(f"Failed to apply edit: {e}")
                else:                
                    print("Edit discarded.")


        elif choice == "4" or choice.lower() == "quit":
            print("Shutting down Aion RAG System. Goodbye!")
            _last_filters.clear()
            break 

        else:
            print("Invalid choice")

    
if __name__ == "__main__":
    asyncio.run(main())
