"""System 1: a plain LLM chatbot.
No tools and no access to the private expense data.
"""

from config import client, MODEL


def chatbot(question):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful student assistant."
            },
            {
                "role": "user",
                "content": question
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":

    print("\n=== SYSTEM 1: CHATBOT ===\n")

    questions = [
        "How much did I spend in total?",
        "Which category had the highest spending?",
        "Did I spend more than Rs. 6,000?"
    ]

    for question in questions:
        print("Q:", question)
        print("A:", chatbot(question))
        print("-" * 70)
