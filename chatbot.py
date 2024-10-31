import random

import gradio as gr
import nltk
from transformers import AutoModelForCausalLM, AutoTokenizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Load the Salesforce CodeGen model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")


def is_greeting(text):
    greetings = ["hi", "hello", "hey", "greetings", "good morning",
                  "good afternoon", "good evening", "yo", "what's up",
                  "howdy", "hiya", "sup", "hello there"]
    return text.lower().strip() in greetings


def generate_code(description, language):
    if is_greeting(description):
        greeting_responses = [
            "Hello there! How can I help you with coding today?",
            "Hi! What coding project are you working on?",
            "Hey! Need some code generated? Let's get started.",
            "Greetings! Tell me about your coding needs."
        ]
        return random.choice(greeting_responses)

    prompt = f"//{language}\n{description}"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated_ids = model.generate(input_ids, max_new_tokens=100)
    generated_code = tokenizer.decode(generated_ids[0])

    # Wrap the generated code in triple backticks for Markdown rendering
    markdown_code = f"```{language}\n{generated_code}\n```"
    return markdown_code


with gr.Blocks(css="styles.css") as iface:  # Use external CSS
    gr.Markdown("# Coding Assisstant")  # Title

    with gr.Column(elem_id="output-panel"):  # Output panel
        output = gr.Markdown(label="Generated Code")

    with gr.Row(elem_id="input-wrapper"):  # Input wrapper
        textbox = gr.Textbox(
            lines=5,
            placeholder="Describe what you want the code to do",
            label="Description")
        dropdown = gr.Dropdown(["Python", "JavaScript", "C++", "Java"],
                             label="Prefered Language")
        button = gr.Button("Enter")

    button.click(generate_code, inputs=[textbox, dropdown], outputs=output)

iface.launch()