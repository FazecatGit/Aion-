from brain.rag_brain import ingest_docs
from brain.augmented_generation import query_brain_comprehensive
from agent.code_agent import CodeAgent


def query_brain_full(question: str, verbose: bool = False) -> dict:
    return query_brain_comprehensive(question, verbose=verbose)


def main():
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
            ingest_docs()
            print("\nIngestion complete. Returning to main menu...")

        elif choice == "2":
            print("\nQuery mode. Type 'quit' to return to main menu.\nType 'verbose' to toggle debug info.\n")
            verbose = False

            while True:
                q = input("Ask: ").strip()
                if q.lower() == "quit":
                    print("Returning to main menu...")
                    break
                
                if q.lower() == "verbose":
                    verbose = not verbose
                    print(f"Verbose mode: {verbose}\n")
                    continue

                if q:
                    results = query_brain_full(q, verbose=verbose)
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
            
            # Ask for repo path (usually '.')
            repo_path = input("Enter the path to your repository (e.g., .): ").strip()
            if repo_path.lower() == "quit":
                continue
                
            try:
                agent = CodeAgent(repo_path)
            except Exception as e:
                print(f"Could not initialize agent: {e}")
                continue

            while True:
                # Ask for the SPECIFIC FILE to edit
                filepath = input("\nEnter specific file to edit (e.g., main.py) or 'quit': ").strip()
                if filepath.lower() == "quit":
                    print("Exiting Code Agent mode. Returning to main menu...")
                    break
                if not filepath:
                    continue

                # Ask for the instruction
                instruction = input("Enter your instruction for the code edit: ").strip()
                if not instruction:
                    continue
                    
                print("\nProcessing your request...\n")

                try:
                    # Pass the filepath, NOT the repo_path, to edit_code
                    agent.edit_code(path=filepath, instruction=instruction, dry_run=True)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    continue

                confirm = input("\nDo you want to apply these changes? (y/n): ").strip().lower()
                if confirm == "y":
                    try:
                        agent.edit_code(path=filepath, instruction=instruction, dry_run=False)
                        print("Edit applied successfully.")
                    except Exception as e:
                        print(f"Failed to apply edit: {e}")
                else:                
                    print("Edit discarded.")


        elif choice == "4" or choice.lower() == "quit":
            print("Shutting down Aion RAG System. Goodbye!")
            break 

        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
