import openai


client = openai.OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="sk-OjjN3nmNeSZxEE8c2QJz985fdY3b9XegsKi7lTcl8z6Sr2de",
    base_url="https://api.chatanywhere.tech/v1"
    # base_url="https://api.chatanywhere.org/v1"
)

class LLMWrapper:
    """
    A unified wrapper for different LLM providers (OpenAI, Anthropic).
    """

    def __init__(self, prompt) -> None:
        self.messages = [{"role": "system", "content": "You are a helpful assistant that generates code and test cases based on user requirements."}]


    def send_message(self, user_input):
        self.messages.append({"role":"user", "content":user_input})
        chat_completion = client.chat.completions.create(
            messages= self.messages,
            model="gpt-3.5-turbo",
        )
        assistant_reply = chat_completion.choices[0].message.content
        self.messages.append({"role":"assistant", "content":assistant_reply})

        return assistant_reply

if __name__ == "__main__":
    prompt = "You are a helpful assistant that generates code and test cases based on user requirements."
    llm = LLMWrapper(prompt)
    response = llm.send_message("Write a function to add two numbers.")
    print(1)
    print(response)

