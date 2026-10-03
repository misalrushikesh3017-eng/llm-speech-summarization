
import os
import tempfile
import time
import streamlit as st
from faster_whisper import WhisperModel
from google import genai
from rouge_score import rouge_scorer

st.set_page_config(
    page_title="Multilingual Speech Summarization",
    page_icon="🎙️"
)

st.title("🎙️ LLM-Based Multilingual Speech Summarization")
st.write("Audio → Speech-to-Text → LLM → Concise Summary")


# ---------- SESSION STATE ----------
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

if "language" not in st.session_state:
    st.session_state.language = ""

if "transcript" not in st.session_state:
    st.session_state.transcript = ""

if "summary1" not in st.session_state:
    st.session_state.summary1 = None

if "summary2" not in st.session_state:
    st.session_state.summary2 = None


# ---------- GEMINI API KEY ----------
api_key = st.text_input(
    "Enter Gemini API Key",
    type="password",
    help="Don't have a Gemini API key? Click here for instructions."
)

with st.expander("🔑 How to get a Gemini API Key?"):
    st.markdown("""
    **Follow these simple steps:**

    1. Open **Google AI Studio**.
    2. Sign in with your Google account.
    3. Click **Get API key**.
    4. Create a new API key.
    5. Copy the API key.
    6. Paste it in the **Gemini API Key** box above.

    ⚠️ **Keep your API key private. Do not share it publicly.**
    """)



# ---------- WHISPER ----------
@st.cache_resource
def load_whisper():
    return WhisperModel(
        "small",
        device="cpu",
        compute_type="int8"
    )


# ---------- GEMINI RETRY ----------
def generate_with_retry(client, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )

        except Exception as e:
            print(f"Gemini Error (Attempt {attempt + 1}): {e}")

            if attempt < max_retries - 1:
                st.warning(
                    f"⏳ Gemini request failed. "
                    f"Retrying... ({attempt + 1}/{max_retries})"
                )
                time.sleep(2)

            else:
                st.error(f"❌ Gemini API Error: {e}")
                return None
    for attempt in range(max_retries):

        try:
            return client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )

        except Exception:

            if attempt < max_retries - 1:
                st.info(
                    f"⏳ Gemini is temporarily busy. "
                    f"Retrying... ({attempt + 1}/{max_retries})"
                )
                time.sleep(2)

            else:
                return None


# ---------- AUDIO UPLOAD ----------
uploaded = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "m4a", "flac"]
)


# ---------- ANALYZE ----------
if uploaded and api_key:

    if st.button("Analyze Audio"):

        client = genai.Client(api_key=api_key)

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(uploaded.name)[1]
        ) as f:
            f.write(uploaded.read())
            audio_path = f.name

        # Speech to Text
        with st.spinner("Transcribing speech..."):

            model = load_whisper()

            segments, info = model.transcribe(
                audio_path,
                beam_size=5
            )

            transcript = " ".join(
                segment.text.strip()
                for segment in segments
            )

        # Save transcription
        st.session_state.language = info.language
        st.session_state.transcript = transcript

        # Prompt 1
        prompt1 = f"""
You are a multilingual speech summarization assistant.

Summarize the following transcript in 3-5 concise sentences.
Preserve the important facts and do not invent information.

Transcript:
{transcript}
"""

        # Prompt 2
        prompt2 = f"""
You are an expert meeting/lecture summarizer.

Read the transcript below and produce:

1. A short 3-5 sentence summary.
2. Three important key points.

Use only information present in the transcript.

Transcript:
{transcript}
"""

        # Generate summaries
        with st.spinner("Generating summaries with Gemini..."):

            r1 = generate_with_retry(client, prompt1)
            r2 = generate_with_retry(client, prompt2)

        if r1:
            st.session_state.summary1 = r1.text
        else:
            st.session_state.summary1 = None

        if r2:
            st.session_state.summary2 = r2.text
        else:
            st.session_state.summary2 = None

        st.session_state.analyzed = True

        os.remove(audio_path)

        st.success("✅ Analysis completed successfully!")


# ---------- SHOW RESULTS ----------
if st.session_state.analyzed:

    st.subheader("1. Detected Language")
    st.write(st.session_state.language)

    st.subheader("2. Transcript")
    st.write(st.session_state.transcript)

    st.subheader("3. Prompt 1 — Concise Summary")

    if st.session_state.summary1:
        st.write(st.session_state.summary1)
    else:
        st.warning(
            "⚠️ Prompt 1 could not be generated. "
            "Please try Analyze Audio again."
        )

    st.subheader("4. Prompt 2 — Summary + Key Points")

    if st.session_state.summary2:
        st.write(st.session_state.summary2)
    else:
        st.warning(
            "⚠️ Prompt 2 could not be generated. "
            "Please try Analyze Audio again."
        )

    st.subheader("5. Prompt Comparison")

    st.table({
        "Feature": [
            "Summary style",
            "Key points",
            "Main purpose"
        ],
        "Prompt 1": [
            "Short 3-5 sentence summary",
            "No",
            "Fast concise summary"
        ],
        "Prompt 2": [
            "Summary + structured points",
            "Yes",
            "Detailed information extraction"
        ]
    })

    # ---------- ROUGE ----------
    st.subheader("6. Optional ROUGE Evaluation")

    reference = st.text_area(
        "Paste a human-written reference summary here (optional):",
        key="reference_summary"
    )

    if reference.strip() and st.session_state.summary1:

        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True
        )

        score = scorer.score(
            reference,
            st.session_state.summary1
        )

        st.write({
            "ROUGE-1 F1": round(
                score["rouge1"].fmeasure, 4
            ),
            "ROUGE-2 F1": round(
                score["rouge2"].fmeasure, 4
            ),
            "ROUGE-L F1": round(
                score["rougeL"].fmeasure, 4
            )
        })

else:

    if not uploaded:
        st.info(
            "Upload an English or Marathi speech recording "
            "and enter your Gemini API key."
        )

    elif not api_key:
        st.warning(
            "Please enter your Gemini API key first."
        )
