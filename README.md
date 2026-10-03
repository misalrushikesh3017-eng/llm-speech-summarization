# LLM-Based Multilingual Speech Summarization

## Project flow
Audio → Whisper (Speech-to-Text) → Transcript → Gemini LLM → Summary

The demo supports English and Marathi speech through Whisper's multilingual transcription.

## What you need
- Google account
- Gemini API key
- Google Colab or a laptop with Python 3.10+
- An English/Marathi audio file (.wav, .mp3, .m4a, .flac)

## Google Colab steps

### 1. Open Colab
Go to https://colab.research.google.com/ and create a new notebook.

### 2. Install packages
Run:
pip install -r requirements.txt

If using Colab, upload requirements.txt first, or run:
!pip install streamlit faster-whisper google-genai rouge-score

### 3. Save app.py
Upload app.py into the Colab Files panel.

### 4. Run Streamlit
In a Colab cell:
!streamlit run app.py &> /content/logs.txt &

Then expose the local Streamlit port using a Colab-compatible tunnel. If your environment does not provide a public URL automatically, run:
!npm install -g localtunnel
!npx localtunnel --port 8501

Open the URL shown by localtunnel.

### 5. Gemini API key
Create a Gemini API key from Google AI Studio and paste it into the app when asked.

### 6. Test
Upload an English or Marathi speech recording.
The app will show:
- detected language
- transcript
- Prompt 1 summary
- Prompt 2 summary
- prompt comparison
- optional ROUGE score

## Creating actual experimental results

For the report, use a small multi-speaker dataset, for example:
- 5 English speakers × 2 recordings = 10 samples
- 5 Marathi speakers × 2 recordings = 10 samples
- Total = 20 recordings

Use consented recordings or an appropriate public dataset.

For each recording, create a short human-written reference summary.
Run the app and record ROUGE-1, ROUGE-2 and ROUGE-L for the Prompt 1 output.
For a fair prompt comparison, modify the code to calculate ROUGE for Prompt 2 as well, or save both outputs and calculate both scores in a separate evaluation notebook.

Do NOT put invented accuracy/results into the paper.

## Important
Whisper's first run downloads the selected model, so the first transcription can take longer.
