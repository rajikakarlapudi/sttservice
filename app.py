import gradio as gr
import whisper
import torch

# Load Whisper model
model = whisper.load_model("large-v3")

if torch.cuda.is_available():
    print("Using GPU for faster inference.")
    model = model.to("cuda")  # Move model to GPU if available

def transcribe_audio(audio_file):
    if audio_file is None:
        return "No file uploaded."

    try:
        # Make a prediction using Whisper
        result = model.transcribe(audio_file, language='en', task='transcribe')
        transcription = result["text"]
        return transcription
    except Exception as e:
        # Provide more informative error messages
        error_message = f"An error occurred during transcription: {str(e)}"
        if "ValueError: could not convert string to float" in str(e):
            error_message += "\n**Tip:** Ensure your audio file is in a supported format (e.g., WAV, FLAC, MP3)."
        return error_message

# Define the Gradio interface
interface = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(type="filepath", label="Upload Audio File"),
    outputs="text",
    title="Speech-to-Text with Whisper Large v3 Model",
    description="Upload an audio file and get the transcribed text using Whisper's large v3 model.\n**Supported formats:** WAV, FLAC, MP3."
)

# Launch the Gradio interface
if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=8000)
