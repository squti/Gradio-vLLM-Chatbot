import gradio as gr
from openai import OpenAI

client = OpenAI(
    base_url="http://vllm:8000/v1",
    api_key="none",
)


def predict(message, history):
    messages = []
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    stream = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=messages,
        max_tokens=8000,
        temperature=0.2,
        stream=True,
    )

    partial_message = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            partial_message += chunk.choices[0].delta.content
            yield partial_message


gr.ChatInterface(
    fn=predict,
    title="Llama-3 3B Chatbot",
    description="Chat with Llama-3-3B-Instruct powered by vLLM",
).launch(server_port=8080, server_name="0.0.0.0", share=False)
