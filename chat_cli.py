"""
Quick terminal client for testing, no frontend needed:

    python chat_cli.py

Type messages; type 'exit' to quit. Uses a fixed demo user_id so memory
persists across runs (change USER_ID to test different "users").
"""

from app import orchestrator

USER_ID = "demo_user"


def main():
    print("Memory-Augmented Chatbot. Type 'exit' to quit.\n")
    while True:
        user_message = input("You: ").strip()
        if user_message.lower() in {"exit", "quit"}:
            break
        if not user_message:
            continue

        result = orchestrator.run_chat(USER_ID, user_message)

        print(f"\nBot [{result.get('route')}]: {result['reply']}")
        sources = sorted(set(c["source"] for c in result.get("rag_chunks") or []))
        sources += sorted(set(m["source"] for m in result.get("graph_matches") or []))
        if sources:
            print(f"(sources: {', '.join(sorted(set(sources)))})\n")
        else:
            print()


if __name__ == "__main__":
    main()
