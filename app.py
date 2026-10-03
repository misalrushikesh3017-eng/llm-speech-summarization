import streamlit as st
import time
import json
from faster_whisper import WhisperModel
from google import genai
from rouge_score import rouge_scorer


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="LLM-Based Multilingual Speech Summarization",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ LLM-Based Multilingual Speech Summarization")
st.write(
    "Upload an audio file to transcribe the speech and generate "
    "an AI-powered summary with key points."
)


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

if "language" not in st.session_state:
    st.session_state.language = ""

if "transcript" not in st.session_state:
    st.session_state.transcript = ""

if "summary1" not in st.session_state:
    st.session_state.summary1 = ""

if "summary2" not in st.session_state:
    st.session_state.summary2 = ""

if "key_points" not in st.session_state:
    st.session_state.key_points = []

if "reference_summary" not in st.session_state:
    st.session_state.reference_summary = ""


# ---------------------------------------------------------
# API KEY SECTION
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# AUDIO UPLOAD
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav", "mp3", "m4a", "flac"]
)


# ---------------------------------------------------------
# GEMINI RETRY FUNCTION
# ---------------------------------------------------------

def generate_with_retry(client, prompt, max_retries=3):

    for attempt in range(max_retries):

        try:

            return client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )

        except Exception as e:

            error_text = str(e)

            print(
                f"Gemini Error (Attempt {attempt + 1}): "
                f"{error_text}"
            )

            # -------------------------------------------------
            # DO NOT RETRY QUOTA ERRORS
            # -------------------------------------------------

            if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:

                st.error(
                    "❌ Gemini API quota has been exceeded. "
                    "Please try again after the quota resets."
                )

                return None

            # -------------------------------------------------
            # RETRY TEMPORARY ERRORS
            # -------------------------------------------------

            if attempt < max_retries - 1:

                st.warning(
                    f"⏳ Gemini request failed. "
                    f"Retrying... ({attempt + 1}/{max_retries})"
                )

                time.sleep(2)

            else:

                st.error(
                    f"❌ Gemini API Error: {error_text}"
                )

                return None

    return None


# ---------------------------------------------------------
# ANALYZE AUDIO
# ---------------------------------------------------------

if uploaded_file and st.button(
    "🚀 Analyze Audio",
    type="primary"
):

    if not api_key:

        st.error(
            "⚠️ Please enter your Gemini API key first."
        )

    else:

        # Reset previous results

        st.session_state.analyzed = False
        st.session_state.language = ""
        st.session_state.transcript = ""
        st.session_state.summary1 = ""
        st.session_state.summary2 = ""
        st.session_state.key_points = []

        # -------------------------------------------------
        # SAVE AUDIO FILE
        # -------------------------------------------------

        audio_path = "/tmp/uploaded_audio"

        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # -------------------------------------------------
        # WHISPER
        # -------------------------------------------------

        with st.spinner(
            "🎧 Transcribing audio..."
        ):

            try:

                model = WhisperModel(
                    "small",
                    device="cpu",
                    compute_type="int8"
                )

                segments, info = model.transcribe(
                    audio_path
                )

                transcript = " ".join(
                    segment.text.strip()
                    for segment in segments
                ).strip()

                language = info.language

                st.session_state.language = language
                st.session_state.transcript = transcript

            except Exception as e:

                st.error(
                    f"❌ Transcription failed: {e}"
                )

                st.stop()

        # -------------------------------------------------
        # CHECK TRANSCRIPT
        # -------------------------------------------------

        if not transcript:

            st.error(
                "❌ No speech could be detected in the audio."
            )

            st.stop()

        # -------------------------------------------------
        # GEMINI CLIENT
        # -------------------------------------------------

        try:

            client = genai.Client(
                api_key=api_key
            )

        except Exception as e:

            st.error(
                f"❌ Gemini client initialization failed: {e}"
            )

            st.stop()

        # -------------------------------------------------
        # SINGLE GEMINI PROMPT
        # -------------------------------------------------

        prompt = f"""
You are an AI multilingual speech summarization system.

The following is a transcript generated from an audio file.

Detected language:
{language}

Transcript:
{transcript}

Perform BOTH tasks below in ONE response:

TASK 1:
Create a concise summary of the transcript in 3–5 sentences.

TASK 2:
Create exactly 3 important key points from the transcript.

Return the result ONLY in this format:

SUMMARY:
<3–5 sentence summary>

KEY POINTS:
1. <key point 1>
2. <key point 2>
3. <key point 3>
"""

        # -------------------------------------------------
        # ONE API CALL ONLY
        # -------------------------------------------------

        with st.spinner(
            "🤖 Generating AI summary..."
        ):

            response = generate_with_retry(
                client,
                prompt
            )

        if response is not None:

            try:

                generated_text = response.text.strip()

                # -------------------------------------------------
                # PARSE SUMMARY
                # -------------------------------------------------

                if "SUMMARY:" in generated_text:

                    summary_part = generated_text.split(
                        "SUMMARY:",
                        1
                    )[1]

                else:

                    summary_part = generated_text

                if "KEY POINTS:" in summary_part:

                    summary_text, key_points_part = (
                        summary_part.split(
                            "KEY POINTS:",
                            1
                        )
                    )

                else:

                    summary_text = summary_part
                    key_points_part = ""

                summary_text = summary_text.strip()

                # -------------------------------------------------
                # PARSE KEY POINTS
                # -------------------------------------------------

                key_points = []

                for line in key_points_part.splitlines():

                    line = line.strip()

                    if not line:
                        continue

                    if line[0].isdigit():

                        if "." in line:

                            point = line.split(
                                ".",
                                1
                            )[1].strip()

                        elif ")" in line:

                            point = line.split(
                                ")",
                                1
                            )[1].strip()

                        else:

                            point = line

                        if point:
                            key_points.append(point)

                # -------------------------------------------------
                # FALLBACK
                # -------------------------------------------------

                if not key_points:

                    key_points = [
                        line.strip("-• ")
                        for line in key_points_part.splitlines()
                        if line.strip("-• ")
                    ]

                # -------------------------------------------------
                # SAVE RESULTS
                # -------------------------------------------------

                st.session_state.summary1 = summary_text

                st.session_state.key_points = (
                    key_points[:3]
                )

                # Summary + Key Points display

                summary2 = summary_text

                if key_points:

                    summary2 += "\n\nKey Points:\n"

                    for i, point in enumerate(
                        key_points[:3],
                        1
                    ):

                        summary2 += (
                            f"{i}. {point}\n"
                        )

                st.session_state.summary2 = (
                    summary2.strip()
                )

                st.session_state.analyzed = True

            except Exception as e:

                st.error(
                    f"❌ Could not process Gemini response: {e}"
                )

        else:

            st.warning(
                "⚠️ AI summary could not be generated. "
                "Your transcription is still available below."
            )


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

if st.session_state.analyzed:

    st.success(
        "✅ Analysis completed successfully!"
    )

    # -----------------------------------------------------
    # LANGUAGE
    # -----------------------------------------------------

    st.subheader("🌐 1. Detected Language")

    st.write(
        st.session_state.language
    )

    # -----------------------------------------------------
    # TRANSCRIPT
    # -----------------------------------------------------

    st.subheader("📝 2. Transcript")

    st.text_area(
        "Transcribed Speech",
        value=st.session_state.transcript,
        height=180,
        disabled=True
    )

    # -----------------------------------------------------
    # PROMPT 1
    # -----------------------------------------------------

    st.subheader(
        "📌 3. Prompt 1 — Concise Summary"
    )

    if st.session_state.summary1:

        st.write(
            st.session_state.summary1
        )

    else:

        st.warning(
            "Prompt 1 could not be generated."
        )

    # -----------------------------------------------------
    # PROMPT 2
    # -----------------------------------------------------

    st.subheader(
        "📌 4. Prompt 2 — Summary + Key Points"
    )

    if st.session_state.summary2:

        st.write(
            st.session_state.summary2
        )

    else:

        st.warning(
            "Prompt 2 could not be generated."
        )

    # -----------------------------------------------------
    # PROMPT COMPARISON
    # -----------------------------------------------------

    if (
        st.session_state.summary1
        and st.session_state.summary2
    ):

        st.subheader(
            "📊 5. Prompt Comparison"
        )

        comparison_data = {

            "Feature": [
                "Summary Length",
                "Key Points",
                "Purpose"
            ],

            "Prompt 1": [
                "3–5 sentences",
                "No",
                "Concise summary"
            ],

            "Prompt 2": [
                "Summary + key points",
                "Yes",
                "Detailed information"
            ]
        }

        st.table(
            comparison_data
        )

    # -----------------------------------------------------
    # ROUGE EVALUATION
    # -----------------------------------------------------

    st.subheader(
        "📈 6. ROUGE Evaluation"
    )

    st.write(
        "Enter a reference summary to compare "
        "the generated summary using ROUGE scores."
    )

    reference_summary = st.text_area(
        "Reference Summary",
        key="reference_summary",
        height=150,
        placeholder="Enter the reference summary here..."
    )

    if st.button(
        "📊 Calculate ROUGE"
    ):

        if not reference_summary.strip():

            st.warning(
                "⚠️ Please enter a reference summary."
            )

        elif not st.session_state.summary1:

            st.warning(
                "⚠️ Generated summary is not available."
            )

        else:

            try:

                scorer = rouge_scorer.RougeScorer(
                    [
                        "rouge1",
                        "rouge2",
                        "rougeL"
                    ],
                    use_stemmer=True
                )

                scores = scorer.score(
                    reference_summary,
                    st.session_state.summary1
                )

                st.write(
                    f"**ROUGE-1 F1:** "
                    f"{scores['rouge1'].fmeasure:.4f}"
                )

                st.write(
                    f"**ROUGE-2 F1:** "
                    f"{scores['rouge2'].fmeasure:.4f}"
                )

                st.write(
                    f"**ROUGE-L F1:** "
                    f"{scores['rougeL'].fmeasure:.4f}"
                )

            except Exception as e:

                st.error(
                    f"❌ ROUGE evaluation failed: {e}"
                )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown("---")

st.caption(
    "LLM-Based Multilingual Speech Summarization "
    "using Faster-Whisper and Gemini"
)
