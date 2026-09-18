import re
import random

# Define rule base: pattern (regular expression) ->
rules = {
    r"I need (.*)": [
        "Why do you need {0}?",
        "Would it really help you to get {0}?",
        "Are you sure you need {0}?",
    ],
    r"Why don\'t you (.*)\?": [
        "Do you really think I don't {0}?",
        "Perhaps eventually I will {0}.",
        "Do you really want me to {0}?",
    ],
    r"Why can\'t I (.*)\?": [
        "Do you think you should be able to {0}?",
        "If you could {0}, what would you do?",
        "I don't know -- why can't you {0}?",
    ],
    r"I am (.*)": [
        "Did you come to me because you are {0}?",
        "How long have you been {0}?",
        "How do you feel about being {0}?",
    ],
    r".* mother .*": [
        "Tell me more about your mother.",
        "What was your relationship with your mother like?",
        "How do you feel about your mother?",
    ],
    r".* father .*": [
        "Tell me more about your father.",
        "How did your father make you feel?",
        "What has your father taught you?",
    ],
    r".*": [
        "Please tell me more.",
        "Let's change focus a bit... Tell me about your family.",
        "Can you elaborate on that?",
    ],
}

# Define pronoun conversion rules
pronouns_swap = {
    "I": "you",
    "me": "you",
    "my": "your",
    "am": "are",
    "you": "I",
    "your": "my",
    "yours": "mine",
    "are": "am",
    "was": "were",
    "were": "was",
    "me": "you",
    "mine": "yours",
    "myself": "yourself",
}


def swap_pronouns(phrase):
    """
    Perform first/second person conversion on pronouns in input phrase
    """
    words = phrase.lower().split()
    swapped_words = [pronouns_swap.get(word, word) for word in words]
    return " ".join(swapped_words)


def respond(user_input: str):
    """
    Generate a response based on rule base
    """
    for pattern, responses in rules.items():
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            # Capture matched part
            caputured_group = match.group(1) if match.group() else ""
            # Perform pronoun conversion
            swapped_group = swap_pronouns(caputured_group)
            # Randomly select one from templates and format
            response = random.choice(responses).format(swapped_group)
            return response
        # If no specific rule is matched, use the last wildcard rule
        return random.choice(rules[r".*"])


# Main chat loop
if __name__ == "__main__":
    print("Therapist: Hello! How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "bye", "exit"]:
            print("Therapist: Goodbye! It was nice talking to you.")
            break
        response = respond(user_input)
        print(f"Therapist: {response}")
