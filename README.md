# JustifAI: The Intelligent Excuse & Apology Generator

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-brightgreen)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**JustifAI** is your personal AI assistant for crafting the perfect excuse or apology, complete with digital "proofs" to back up your story.

This isn't just a text generator; it's a sophisticated tool that leverages a fine-tuned language model to create context-aware, plausible responses and generates supporting evidence like fake WhatsApp screenshots and voice notes.


---

### ✨ Core Features

*   **🧠 Intelligent Generation**: Uses a PEFT-fine-tuned model based on `mistralai/Mistral-7B-Instruct-v0.2` for high-quality, context-aware excuses and apologies.
*   **📝 Customizable Input**: Tailor your message by specifying the situation, priority, plausibility, and additional context.
*   **🔬 Multi-Format Proofs**:
    *   **📱 WhatsApp Screenshot**: Generates a realistic WhatsApp screenshot of your excuse and a random reply.
    *   **🎤 Voice Note**: Creates an MP3 voice note of the generated message using text-to-speech.
    *   **📍 Location Context**: The AI suggests a plausible location-based scenario to support your story.
*   **💡 Adaptive Suggestions**: Learns from your feedback! Previously rated excuses are suggested for similar future situations.
*   **📜 History & Feedback**: All generations are saved in a history log where you can rate their effectiveness and mark favorites.
*   **📁 Data Management**: Easily load your own excuse/apology datasets and save/load your generation history.

---

### 🤖 The AI Model

The core of JustifAI is powered by one of the leading open-source language models, enhanced with custom fine-tuning.

*   **Base Model**: **[`mistralai/Mistral-7B-Instruct-v0.2`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)**, chosen for its exceptional balance of performance and efficiency.
*   **Customization**: **PEFT Fine-Tuning** adapters are loaded on top of the base model. This means the model has been further trained on a specialized dataset, honing its ability to match the required tone and context with great accuracy.

---

### 🛠️ Tech Stack

*   **Frontend**: **[Streamlit](https://streamlit.io/)**
*   **Core AI & ML**:
    *   **[PyTorch](https://pytorch.org/)**: The deep learning framework that powers the model.
    *   **[Hugging Face Transformers](https://huggingface.co/docs/transformers/index)**: For loading the model and tokenizer.
    *   **[Hugging Face PEFT](https://github.com/huggingface/peft)**: For loading the fine-tuning adapters.
    *   **[Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index)**: For optimizing model performance across different hardware.
*   **Proof Generation**:
    *   **[Pillow (PIL)](https://python-pillow.org/)**: For creating the WhatsApp screenshot images.
    *   **[gTTS (Google Text-to-Speech)](https://github.com/pndurette/gTTS)**: For generating the voice note audio.
*   **Data Handling**: **[Pandas](https://pandas.pydata.org/)** & **[NumPy](https://numpy.org/)**

