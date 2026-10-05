import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is missing. Add it to your .env file.")

# Create Hugging Face client
client = InferenceClient(
    model="google/flan-t5-base",
    token=HF_TOKEN
)


def generate_flashcards(text, number_of_cards=5):
    prompt = f"""
Create {number_of_cards} educational flashcards from the text below.

Rules:
- Each flashcard must have a Question and Answer.
- Questions should test important concepts.
- Answers should be short and accurate.
- Do not add unnecessary explanations.

Text:
{text}

Format:
Q1: question
A1: answer

Q2: question
A2: answer
"""

    response = client.text_generation(
        prompt,
        max_new_tokens=500,
        temperature=0.7
    )

    return response


def display_flashcards(result):
    print("\n" + "=" * 50)
    print("             GENERATED FLASHCARDS")
    print("=" * 50)

    print(result)

    print("=" * 50)


def main():
    print("📚 AI Flashcard Generator")
    print("-" * 40)

    text = input("\nEnter your study text:\n")

    if not text.strip():
        print("Please enter some text.")
        return

    try:
        result = generate_flashcards(text, 5)
        display_flashcards(result)

    except Exception as e:
        print("\n❌ Error:", e)


if __name__ == "__main__":
    main()