# app.py
import streamlit as st
import pandas as pd
import numpy as np # For np.nan
import os
import random
from datetime import datetime
import tempfile

# Import functions from your logic file
from excuse_generator_logic import (
    initialize_model_and_tokenizer,
    load_dataframes_from_objects,
    generate_response,
    generate_voice_output_st,
    generate_whatsapp_screenshot_st,
    generate_location_context,
    trigger_fake_emergency,
    add_to_history_df,
    save_history_to_csv,
    toggle_favorite_in_df,
    record_feedback_in_df,
    get_ranked_suggestion_list,
    BASE_MODEL_ID,
    FINE_TUNED_ADAPTER_DIR,
    RECIPIENT_REPLIES,
    DEFAULT_FONT_SIZE_MSG,
    DEFAULT_FONT_SIZE_INFO,
    HISTORY_DF_COLUMNS
)

# --- Page Configuration ---
st.set_page_config(page_title="Excuse Generator AI", layout="wide", initial_sidebar_state="expanded")

# --- Session State Initialization ---
# These should be light and not trigger heavy computation on import
if 'model' not in st.session_state: st.session_state.model = None
if 'tokenizer' not in st.session_state: st.session_state.tokenizer = None
if 'device' not in st.session_state: st.session_state.device = "cpu" # Default, will be updated
if 'is_model_fine_tuned' not in st.session_state: st.session_state.is_model_fine_tuned = False
if 'font_path_streamlit' not in st.session_state: st.session_state.font_path_streamlit = None
if 'combined_df_for_context' not in st.session_state: st.session_state.combined_df_for_context = pd.DataFrame()
if 'history_df' not in st.session_state: st.session_state.history_df = pd.DataFrame(columns=HISTORY_DF_COLUMNS)
if 'history_id_counter' not in st.session_state: st.session_state.history_id_counter = 0
if 'last_generation' not in st.session_state: st.session_state.last_generation = {}
if 'ui_messages' not in st.session_state: st.session_state.ui_messages = []
if 'hf_token_sidebar_input' not in st.session_state: st.session_state.hf_token_sidebar_input = ""


# Temporary directory for proofs - create it if it doesn't exist
# This will be created in the Streamlit Cloud environment's temporary space
try:
    TEMP_PROOF_DIR = tempfile.mkdtemp(prefix="excusegen_proofs_")
except Exception as e:
    # Fallback if mkdtemp fails (e.g., permission issues in some restricted environments, though unlikely for Streamlit Cloud)
    TEMP_PROOF_DIR = "temp_proofs" # Relative path
    if not os.path.exists(TEMP_PROOF_DIR):
        try:
            os.makedirs(TEMP_PROOF_DIR)
        except Exception as e_mkdir:
            st.error(f"Failed to create temporary directory: {e_mkdir}. Proofs might not work.")
            TEMP_PROOF_DIR = "." # Current directory as last resort

# --- Helper Functions for UI ---
def add_ui_message(message, type="info"):
    st.session_state.ui_messages.append({"message": message, "type": type, "time": datetime.now()})
    # Keep only the last few messages to prevent clutter
    st.session_state.ui_messages = st.session_state.ui_messages[-5:]


def display_ui_messages():
    # Display only recent messages
    messages_to_display = st.session_state.ui_messages
    for msg_info in messages_to_display:
        if msg_info["type"] == "error": st.error(msg_info["message"])
        elif msg_info["type"] == "warning": st.warning(msg_info["message"])
        elif msg_info["type"] == "success": st.success(msg_info["message"])
        else: st.info(msg_info["message"])
    # Clear messages after displaying them once, or implement a more sophisticated clearing logic
    # For now, let's keep them until new ones push them out due to the limit in add_ui_message

# --- Sidebar for Setup & Configuration ---
with st.sidebar:
    st.header("⚙️ Setup & Configuration")
    st.markdown("---")

    # Hugging Face Token - Store input in session state for potential use if secrets fail or for override
    st.session_state.hf_token_sidebar_input = st.text_input(
        "Hugging Face Token (Optional - for local use or override)",
        type="password",
        help="For Streamlit Cloud, set HUGGING_FACE_TOKEN in Secrets.",
        value=st.session_state.hf_token_sidebar_input # Persist value across reruns
    )

    st.markdown("---")
    st.subheader("📤 Upload Files")
    uploaded_excuse_csv = st.file_uploader("Excuse Dataset (CSV)", type="csv", key="excuse_csv")
    uploaded_apology_csv = st.file_uploader("Apology Dataset (CSV)", type="csv", key="apology_csv")
    uploaded_history_csv = st.file_uploader("Load Previous History (CSV, Optional)", type="csv", key="history_csv_upload")
    uploaded_font_file = st.file_uploader("Custom Font File (.ttf, Optional)", type="ttf", key="font_upload")

    if st.button("🔄 Process Uploaded Files", key="process_files_btn"):
        with st.spinner("Processing files..."):
            if uploaded_font_file:
                # Ensure TEMP_PROOF_DIR is writable
                font_save_dir = TEMP_PROOF_DIR
                if not os.access(font_save_dir, os.W_OK):
                    # Fallback if default temp dir isn't writable (unlikely on Streamlit Cloud but good for robustness)
                    font_save_dir = "."
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".ttf", dir=font_save_dir) as tmp_font:
                        tmp_font.write(uploaded_font_file.getvalue())
                        st.session_state.font_path_streamlit = tmp_font.name
                    add_ui_message(f"Font '{uploaded_font_file.name}' ready.", "success")
                except Exception as e:
                    add_ui_message(f"Error saving uploaded font: {e}", "error")
                    st.session_state.font_path_streamlit = None
            else:
                st.session_state.font_path_streamlit = None
                add_ui_message("No custom font uploaded. Default will be used for images.", "info")

            # Load datasets and history
            st.session_state.combined_df_for_context, st.session_state.history_df, st.session_state.history_id_counter = \
                load_dataframes_from_objects(uploaded_excuse_csv, uploaded_apology_csv, uploaded_history_csv)

            if not st.session_state.combined_df_for_context.empty:
                add_ui_message(f"Excuse/Apology datasets loaded: {len(st.session_state.combined_df_for_context)} entries.", "success")
            else:
                add_ui_message("No excuse/apology data loaded. Context might be limited.", "warning")

            if 'id' in st.session_state.history_df.columns and not st.session_state.history_df.empty:
                 add_ui_message(f"History loaded: {len(st.session_state.history_df)} entries. Next ID: {st.session_state.history_id_counter}", "success")
            elif uploaded_history_csv : # If a history file was uploaded but resulted in an empty df (or df without id)
                 add_ui_message(f"History file processed. Next ID: {st.session_state.history_id_counter}", "info")


    st.markdown("---")
    # Model initialization is now strictly behind this button
    if st.button("🚀 Initialize AI Model", key="init_model_btn"):
        if st.session_state.model is None:
            with st.spinner("Initializing AI model... This may take a few minutes on first load."):
                # Prioritize token from Streamlit secrets, then sidebar input
                hf_token_for_init = None
                try:
                    hf_token_for_init = st.secrets.get("HUGGING_FACE_TOKEN")
                    if hf_token_for_init:
                        add_ui_message("Using Hugging Face token from Streamlit Secrets.", "info")
                except Exception: # st.secrets might not exist in all local environments
                    add_ui_message("Streamlit Secrets not available (expected for local runs without secrets.toml).", "info")

                if not hf_token_for_init and st.session_state.hf_token_sidebar_input:
                    hf_token_for_init = st.session_state.hf_token_sidebar_input
                    add_ui_message("Using Hugging Face token from sidebar input.", "info")
                
                if not hf_token_for_init:
                    add_ui_message("No Hugging Face token provided via Secrets or sidebar. Attempting public access.", "warning")

                model, tokenizer, is_ft, device = initialize_model_and_tokenizer(
                    model_id=BASE_MODEL_ID,
                    adapter_dir=FINE_TUNED_ADAPTER_DIR,
                    hf_token=hf_token_for_init
                )
                if model and tokenizer:
                    st.session_state.model = model
                    st.session_state.tokenizer = tokenizer
                    st.session_state.is_model_fine_tuned = is_ft
                    st.session_state.device = device
                    add_ui_message(f"AI Model initialized on {device}. Fine-tuned: {is_ft}", "success")
                else:
                    add_ui_message("AI Model initialization FAILED. Check logs.", "error")
                    # Ensure session state reflects failure
                    st.session_state.model = None
                    st.session_state.tokenizer = None
        else:
            add_ui_message(f"AI Model already initialized on {st.session_state.device}. Fine-tuned: {st.session_state.is_model_fine_tuned}", "info")

    st.markdown("---")
    # Display model status
    if st.session_state.model and st.session_state.tokenizer:
        status_msg = f"✅ Model Ready ({st.session_state.device})"
        if st.session_state.is_model_fine_tuned:
            status_msg += " (Fine-tuned)"
        st.success(status_msg)
    else:
        st.warning("Model not initialized. Click 'Initialize AI Model'.")

    if not st.session_state.combined_df_for_context.empty:
        st.success("✅ Context Data Loaded")
    else:
        st.info("Context data (excuse/apology CSVs) not loaded.")


# --- Main Application Area ---
st.title("🧠 Intelligent Excuse & Apology Generator")
display_ui_messages() # Display status messages from sidebar actions

if not st.session_state.model or not st.session_state.tokenizer:
    st.warning("👈 Please initialize the AI model using the button in the sidebar first.")
    st.caption("Note: Model initialization may take a few minutes, especially on the first run or on free cloud tiers.")
else:
    # All main app logic that uses the model goes here
    tab1, tab2, tab3 = st.tabs(["💬 Generate New", "📜 History", "⚡ Actions"])

    with tab1:
        st.header("📝 Generate New Message")
        with st.form("generation_form"):
            c1, c2 = st.columns(2)
            with c1:
                situation = st.text_input("Situation (e.g., 'late for meeting')", key="situation_input")
                priority = st.selectbox("Priority", ["low", "medium", "high"], key="priority_input", index=1)
            with c2:
                plausibility = st.selectbox("Plausibility", ["low", "medium", "high"], key="plausibility_input", index=1)
                msg_type = st.selectbox("Message Type", ["excuse", "apology"], key="msg_type_input")

            user_context = st.text_area("Additional Context/Reason (be specific for better results)", key="user_context_input", height=100)
            submit_button = st.form_submit_button("✨ Generate Message")

        if submit_button:
            if not situation: # Removed user_context check to allow generation without it
                add_ui_message("'Situation' cannot be empty.", "error")
            else:
                with st.spinner("AI is thinking..."):
                    # 1. Get Ranked Suggestions
                    ranked_suggestions = get_ranked_suggestion_list(
                        st.session_state.history_df, situation, priority, plausibility, msg_type
                    )
                    if ranked_suggestions:
                        st.subheader("💡 Previously Rated Suggestions")
                        for sug in ranked_suggestions:
                            st.markdown(f"- \"{sug['text']}\" *(Rated: {sug['rating']})*")
                        st.markdown("---")

                    # 2. Generate New Message
                    # Construct prompt based on inputs
                    prompt_instruction = (
                        f"Generate a short, informal text message style {msg_type} for the situation: '{situation}'. "
                        f"The specific reason or context is: '{user_context}'. "
                        f"The message should sound like it has {plausibility} plausibility and {priority} urgency. "
                        f"Keep it concise (1-2 sentences typically). Avoid formal greetings or sign-offs. "
                        f"Output ONLY the message text itself."
                    )
                    generated_text = generate_response(st.session_state.model, st.session_state.tokenizer, prompt_instruction, st.session_state.device)

                    if generated_text and not generated_text.startswith("Error:") and generated_text != "[Empty Response]":
                        current_id_for_history = st.session_state.history_id_counter
                        st.session_state.last_generation = {
                            "id": current_id_for_history,
                            "situation": situation, "priority": priority, "plausibility": plausibility,
                            "msg_type": msg_type, "user_context": user_context, "generated_text": generated_text,
                            "timestamp": datetime.now() # Add timestamp to last_generation for consistency
                        }
                        # Add to history
                        history_entry_data = {
                            'timestamp': st.session_state.last_generation["timestamp"], # Use consistent timestamp
                            'situation': situation, 'priority': priority,
                            'plausibility': plausibility, 'message_type': msg_type, 'user_context': user_context,
                            'generated_text': generated_text, 'effectiveness_rating': np.nan, 'is_favorite': False
                        }
                        st.session_state.history_df, st.session_state.history_id_counter = add_to_history_df(
                            st.session_state.history_df, history_entry_data, current_id_for_history
                        )
                        add_ui_message(f"Generated new {msg_type} (ID: {current_id_for_history}).", "success")

                        # 3. Generate Proofs
                        st.session_state.last_generation["audio_file_path"] = generate_voice_output_st(generated_text, output_dir=TEMP_PROOF_DIR)
                        st.session_state.last_generation["whatsapp_img_path"] = generate_whatsapp_screenshot_st(
                            generated_text, random.choice(RECIPIENT_REPLIES),
                            st.session_state.font_path_streamlit,
                            DEFAULT_FONT_SIZE_MSG, DEFAULT_FONT_SIZE_INFO, output_dir=TEMP_PROOF_DIR
                        )
                        st.session_state.last_generation["location_context"] = generate_location_context(
                            st.session_state.model, st.session_state.tokenizer, generated_text, user_context, st.session_state.device
                        )
                    else:
                        add_ui_message(f"Message generation failed: {generated_text}", "error")
                        st.session_state.last_generation = {} # Clear previous if failed
            # Display UI messages from generation attempt
            display_ui_messages()

        # --- Display Last Generation Results & Feedback ---
        if st.session_state.last_generation.get("generated_text"):
            lg = st.session_state.last_generation
            st.divider()
            st.subheader(f"💬 Generated {lg['msg_type'].capitalize()} (ID: {lg['id']})")
            st.markdown(f"##### **Message:**\n> {lg['generated_text']}")

            proof_cols = st.columns(3)
            with proof_cols[0]:
                if lg.get("audio_file_path") and os.path.exists(lg["audio_file_path"]):
                    st.caption("🎤 Voice Note:")
                    try:
                        with open(lg["audio_file_path"], "rb") as audio_f:
                            st.audio(audio_f.read(), format='audio/mp3')
                    except Exception as e_audio:
                        st.caption(f"🎤 Voice Note: (Error displaying: {e_audio})")
                else: st.caption("🎤 Voice Note: (failed or N/A)")
            with proof_cols[1]:
                if lg.get("whatsapp_img_path") and os.path.exists(lg["whatsapp_img_path"]):
                    st.caption("📱 WhatsApp Proof:")
                    st.image(lg["whatsapp_img_path"], width=300)
                else: st.caption("📱 WhatsApp Proof: (failed or N/A)")
            with proof_cols[2]:
                if lg.get("location_context"):
                    st.caption("📍 Location Context:")
                    st.text_area("", value=lg["location_context"], height=100, disabled=True, key=f"loc_ctx_{lg['id']}_{random.randint(0,10000)}") # Unique key
                else: st.caption("📍 Location Context: (failed or N/A)")

            st.markdown("---")
            # Feedback section
            if 'id' in lg and isinstance(st.session_state.history_df, pd.DataFrame) and lg['id'] in st.session_state.history_df['id'].values:
                history_item_index = st.session_state.history_df.index[st.session_state.history_df['id'] == lg['id']].tolist()
                if history_item_index:
                    idx = history_item_index[0]
                    st.markdown("**Rate this generation (ID: {}):**".format(lg['id']))
                    fb_cols = st.columns([1,3,1.5])

                    with fb_cols[0]:
                        current_fav_status = bool(st.session_state.history_df.loc[idx, 'is_favorite'])
                        if st.button("❤️" if current_fav_status else "🤍", key=f"fav_btn_{lg['id']}_{random.randint(0,10000)}", help="Toggle Favorite"):
                            st.session_state.history_df = toggle_favorite_in_df(st.session_state.history_df, lg['id'])
                            add_ui_message(f"Favorite status updated for ID {lg['id']}.", "info")
                            st.rerun()

                    with fb_cols[1]:
                        current_rating_val = st.session_state.history_df.loc[idx, 'effectiveness_rating']
                        if pd.isna(current_rating_val): current_rating_val = 5 # Default if NaN
                        rating_input = st.slider("Effectiveness (0-10)", 0, 10, value=int(current_rating_val), key=f"rate_slider_{lg['id']}_{random.randint(0,10000)}")

                    with fb_cols[2]:
                        st.write("") # Spacer
                        if st.button("Submit Rating", key=f"submit_rating_btn_{lg['id']}_{random.randint(0,10000)}"):
                            st.session_state.history_df = record_feedback_in_df(st.session_state.history_df, lg['id'], rating_input)
                            add_ui_message(f"Rating {rating_input}/10 saved for ID {lg['id']}.", "success")
                            st.rerun()
                else:
                    st.warning(f"Could not find history item ID {lg['id']} for feedback.")
            else:
                 st.info(f"Feedback can be provided once item ID {lg.get('id')} is confirmed in history.")
            display_ui_messages()


    with tab2:
        st.header("📜 Generation History")
        if not st.session_state.history_df.empty:
            history_display_df = st.session_state.history_df.copy()
            # Ensure correct formatting for display
            if 'timestamp' in history_display_df.columns:
                history_display_df['timestamp'] = pd.to_datetime(history_display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            if 'effectiveness_rating' in history_display_df.columns:
                history_display_df['effectiveness_rating'] = history_display_df['effectiveness_rating'].apply(lambda x: f"{int(x)}/10" if pd.notna(x) else "N/A")
            if 'is_favorite' not in history_display_df.columns: # Add if missing
                history_display_df['is_favorite'] = False

            display_columns = ['id', 'timestamp', 'situation', 'message_type', 'generated_text', 'effectiveness_rating', 'is_favorite']
            # Filter to only columns that actually exist in the dataframe to prevent KeyErrors
            actual_display_columns = [col for col in display_columns if col in history_display_df.columns]

            st.dataframe(history_display_df[actual_display_columns].sort_values(by="timestamp", ascending=False), height=400)

            # Save history button and download
            if st.button("💾 Save Current History to CSV", key="save_hist_btn"):
                # Ensure TEMP_PROOF_DIR is writable or use fallback
                save_dir = TEMP_PROOF_DIR
                if not os.access(save_dir, os.W_OK): save_dir = "."

                # Generate a unique filename for the history CSV
                history_file_name = f"excuse_generator_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                saved_path = os.path.join(save_dir, history_file_name)

                if save_history_to_csv(st.session_state.history_df, saved_path):
                    add_ui_message(f"History prepared for download as {history_file_name}", "success")
                    try:
                        with open(saved_path, "rb") as fp:
                            st.download_button(
                                label="⬇️ Download History CSV",
                                data=fp,
                                file_name=history_file_name, # Use just the filename
                                mime="text/csv",
                                key="download_hist_btn"
                            )
                        # Optionally remove the temp file after download setup if it's truly temporary
                        # os.unlink(saved_path) # Be careful with this if download is async
                    except Exception as e_download:
                        add_ui_message(f"Error preparing download: {e_download}", "error")

                else:
                    add_ui_message("Failed to save history to CSV.", "error")
        else:
            st.info("No history yet. Generate some messages!")
        display_ui_messages()

    with tab3:
        st.header("⚡ Other Actions")
        if st.button("🚨 Trigger Fake Emergency Simulation", key="emergency_btn"):
            with st.spinner("Simulating emergency..."):
                emergency_text = trigger_fake_emergency(st.session_state.model, st.session_state.tokenizer, st.session_state.device)
                st.info(emergency_text) # Display simulation result
                add_ui_message("Fake emergency simulation triggered.", "info")

        st.markdown("---")
        st.subheader("🗑️ Clear Temporary Proof Files")
        st.caption(f"Proof files are stored temporarily in: {TEMP_PROOF_DIR}")
        if st.button("Clear Proofs Directory", help="Deletes files in the temporary proofs directory."):
            cleared_count = 0
            try:
                if os.path.exists(TEMP_PROOF_DIR) and os.path.isdir(TEMP_PROOF_DIR):
                    for filename in os.listdir(TEMP_PROOF_DIR):
                        file_path = os.path.join(TEMP_PROOF_DIR, filename)
                        try:
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                                cleared_count +=1
                        except Exception as e_file_del:
                            add_ui_message(f"Could not delete {file_path}: {e_file_del}", "warning")
                    add_ui_message(f"Cleared {cleared_count} files from temporary proofs directory.", "success")
                else:
                    add_ui_message("Temporary proofs directory does not exist or is not a directory.", "warning")
            except Exception as e:
                add_ui_message(f"Error clearing proofs directory: {e}", "error")
        display_ui_messages()


# --- Footer ---
st.markdown("---")
st.caption(f"Excuse Generator AI - Alpha Version. Temp dir: {TEMP_PROOF_DIR}")