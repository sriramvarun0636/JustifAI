# -*- coding: utf-8 -*-
import streamlit as st
import torch
import pandas as pd
import numpy as np
import random
import os
import json
from datetime import datetime
import textwrap # For better text wrapping in images
import re # For cleaning generated text
import traceback # For detailed error printing

# Hugging Face Libraries
from datasets import Dataset # Keep if used in future or by dependencies
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    logging as hf_logging
)
from peft import PeftModel # Keep if used with fine-tuned adapters
# from trl import SFTTrainer # Not used in this interactive app version

# Proof Generation & Utilities
from gtts import gTTS # For Text-to-Speech
from PIL import Image, ImageDraw, ImageFont

from huggingface_hub import login as hf_login_cli
# from huggingface_hub import HfFolder # Not strictly needed for token-based login

# Suppress excessive Hugging Face warnings
hf_logging.set_verbosity_error() # Stricter for cleaner Streamlit output

# --- Constants & Configuration ---
BASE_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
FINE_TUNED_ADAPTER_DIR = "phi3-mini-excuse-generator-adapter"
EXCUSE_CSV_PATH = 'excuse_dataset_cleaned_FIXED.csv'
APOLOGY_CSV_PATH = 'apology_dataset_template_FIXED.csv'
HISTORY_CSV_PATH = 'excuse_dataset_mixed.csv'
GENERATED_AUDIO_PATH = 'generated_output.mp3'
GENERATED_WHATSAPP_PATH = 'generated_whatsapp.png'
UPLOADED_FONT_NAME = "Tahoma.ttf"
FONT_PATH = UPLOADED_FONT_NAME # Assumes font is in the same directory
DEFAULT_FONT_SIZE_MSG = 16
DEFAULT_FONT_SIZE_INFO = 11
RECIPIENT_REPLIES = ["Okay", "Alright, thanks", "Got it.", "Ok, hope you're alright?", "Understood.", "👍", "Sure thing.", "Ok.", "No worries.", "Seen."]

# --- Initialize Streamlit Session State ---
if 'history_df' not in st.session_state:
    st.session_state.history_df = pd.DataFrame(columns=['timestamp', 'situation', 'priority', 'plausibility', 'message_type', 'user_context', 'generated_text', 'effectiveness_rating', 'is_favorite'])
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'is_model_fine_tuned' not in st.session_state:
    st.session_state.is_model_fine_tuned = False
if 'gpu_available' not in st.session_state:
    st.session_state.gpu_available = torch.cuda.is_available()
if 'load_on_cpu' not in st.session_state:
    st.session_state.load_on_cpu = not st.session_state.gpu_available
if 'current_generated_text' not in st.session_state:
    st.session_state.current_generated_text = None
if 'current_history_index' not in st.session_state:
    st.session_state.current_history_index = None
if 'font_available' not in st.session_state:
    st.session_state.font_available = os.path.exists(FONT_PATH)
if 'app_initialized' not in st.session_state:
    st.session_state.app_initialized = False


# --- Helper Functions (some modified for Streamlit) ---

# Section 2: Data Loading and Preparation
@st.cache_data # Cache data loading
def load_datasets():
    excuse_original_cols = ['scenario', 'urgency', 'believability', 'excuse']
    apology_original_cols = ['scenario', 'urgency', 'believability', 'apology']
    standard_cols_map = {
        'scenario': 'situation', 'urgency': 'priority', 'believability': 'plausibility',
        'excuse': 'message_text', 'apology': 'message_text'
    }
    standard_cols_final = ['situation', 'priority', 'plausibility', 'message_text']
    combined_df = pd.DataFrame() # Initialize empty

    try:
        if not os.path.exists(EXCUSE_CSV_PATH):
            st.error(f"Excuse file not found: {EXCUSE_CSV_PATH}")
            return None
        if not os.path.exists(APOLOGY_CSV_PATH):
            st.error(f"Apology file not found: {APOLOGY_CSV_PATH}")
            return None

        excuse_df = pd.read_csv(EXCUSE_CSV_PATH)
        apology_df = pd.read_csv(APOLOGY_CSV_PATH)

        missing_excuse_cols = [col for col in excuse_original_cols if col not in excuse_df.columns]
        if missing_excuse_cols:
            st.error(f"Excuse CSV missing original columns: {missing_excuse_cols}")
            return None
        missing_apology_cols = [col for col in apology_original_cols if col not in apology_df.columns]
        if missing_apology_cols:
            st.error(f"Apology CSV missing original columns: {missing_apology_cols}")
            return None
        
        excuse_df.dropna(subset=excuse_original_cols, inplace=True)
        apology_df.dropna(subset=apology_original_cols, inplace=True)
        excuse_df = excuse_df[excuse_original_cols].copy()
        excuse_df.rename(columns=standard_cols_map, inplace=True)
        excuse_df['message_type'] = 'excuse'
        apology_df = apology_df[apology_original_cols].copy()
        apology_df.rename(columns=standard_cols_map, inplace=True)
        apology_df['message_type'] = 'apology'

        combined_df = pd.concat([excuse_df[standard_cols_final + ['message_type']], apology_df[standard_cols_final + ['message_type']]], ignore_index=True)
        combined_df = combined_df.astype(str)
        st.success(f"Datasets loaded and combined: {len(combined_df)} total valid entries.")
        return combined_df

    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        # traceback.print_exc() # For debugging, prints to console
        return None

def load_history_df():
    history_cols_expected = ['timestamp', 'situation', 'priority', 'plausibility', 'message_type', 'user_context', 'generated_text', 'effectiveness_rating', 'is_favorite']
    if os.path.exists(HISTORY_CSV_PATH):
        try:
            history_df_loaded = pd.read_csv(HISTORY_CSV_PATH, low_memory=False)
            missing_hist_cols = [col for col in history_cols_expected if col not in history_df_loaded.columns]
            if missing_hist_cols:
                for col in missing_hist_cols: history_df_loaded[col] = None
            
            loaded_df = history_df_loaded[history_cols_expected].copy()
            loaded_df['timestamp'] = pd.to_datetime(loaded_df['timestamp'], errors='coerce')
            loaded_df['effectiveness_rating'] = pd.to_numeric(loaded_df['effectiveness_rating'], errors='coerce')
            loaded_df['is_favorite'] = loaded_df['is_favorite'].apply(lambda x: str(x).strip().lower() == 'true' if pd.notna(x) else False).astype(bool)
            loaded_df.dropna(subset=['timestamp'], inplace=True) # Drop rows with invalid timestamps
            st.session_state.history_df = loaded_df
            st.info(f"Loaded {len(st.session_state.history_df)} history entries.")
        except Exception as e:
            st.warning(f"Error loading history: {e}. Starting with empty history.")
            st.session_state.history_df = pd.DataFrame(columns=history_cols_expected)
    else:
        st.info("No history file found. Starting with empty history.")
        st.session_state.history_df = pd.DataFrame(columns=history_cols_expected)


# Section 4: Model Loading for Inference
@st.cache_resource # Cache model and tokenizer loading
def load_model_and_tokenizer(is_fine_tuned_adapter_present, load_on_cpu_flag):
    model_loaded = None
    tokenizer_loaded = None
    
    bnb_config_inference = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False
    )

    effective_load_on_cpu = load_on_cpu_flag or not st.session_state.gpu_available

    if not effective_load_on_cpu:
        st.write(f"Attempting GPU load of base model: {BASE_MODEL_ID}...")
        try:
            model_loaded = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                quantization_config=bnb_config_inference,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            st.write("Base model loaded onto GPU (quantized).")

            if is_fine_tuned_adapter_present and os.path.isdir(FINE_TUNED_ADAPTER_DIR): # Check directory exists
                st.write(f"Attempting to load fine-tuned PEFT adapter from {FINE_TUNED_ADAPTER_DIR}...")
                try:
                    model_loaded = PeftModel.from_pretrained(model_loaded, FINE_TUNED_ADAPTER_DIR)
                    st.write("Fine-tuned PEFT adapter loaded successfully.")
                    st.session_state.is_model_fine_tuned = True
                except Exception as e:
                    st.warning(f"Failed to load PEFT adapter: {e}. Using base model only.")
                    st.session_state.is_model_fine_tuned = False # Explicitly set to false on failure
            else:
                if is_fine_tuned_adapter_present: # If it was expected but not found
                     st.warning(f"Fine-tuned adapter directory '{FINE_TUNED_ADAPTER_DIR}' not found. Using base model.")
                else:
                    st.write("Using base model (no fine-tuning adapter specified or found).")
                st.session_state.is_model_fine_tuned = False

        except Exception as e:
            st.error(f"GPU model load failed: {e}. Falling back to CPU if possible.")
            # traceback.print_exc() # Uncomment for detailed console error
            st.session_state.load_on_cpu = True # Update state to force CPU next time (if applicable)
            effective_load_on_cpu = True # Ensure we try CPU path now
            model_loaded = None # Reset
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    if effective_load_on_cpu and model_loaded is None: # If GPU failed or was intended for CPU
        st.write(f"Attempting CPU load of base model: {BASE_MODEL_ID} (This will be VERY SLOW)...")
        st.session_state.is_model_fine_tuned = False # Adapters usually not simple with CPU-only load this way
        try:
            model_loaded = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
            st.write("Base model loaded onto CPU.")
        except Exception as e:
            st.error(f"FATAL: CPU model load failed: {e}")
            # traceback.print_exc() # Uncomment for detailed console error
            return None, None # Critical failure

    if model_loaded:
        st.write(f"Loading tokenizer for {BASE_MODEL_ID}...")
        try:
            tokenizer_loaded = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
            if tokenizer_loaded.pad_token is None:
                if tokenizer_loaded.eos_token:
                    tokenizer_loaded.pad_token = tokenizer_loaded.eos_token
                    st.write("Set tokenizer pad_token to eos_token.")
                else:
                    # This case should be rare for Phi-3, but as a fallback:
                    tokenizer_loaded.add_special_tokens({'pad_token': '[PAD]'})
                    model_loaded.resize_token_embeddings(len(tokenizer_loaded)) # Important
                    st.write("Added new PAD token as tokenizer had no pad_token or eos_token.")
            tokenizer_loaded.padding_side = "left" # Important for decoder-only
            st.write(f"Tokenizer loaded. Padding side: {tokenizer_loaded.padding_side}. Pad token ID: {tokenizer_loaded.pad_token_id}")
        except Exception as e:
            st.error(f"ERROR loading tokenizer: {e}")
            # traceback.print_exc() # Uncomment for detailed console error
            tokenizer_loaded = None
    
    return model_loaded, tokenizer_loaded


# Section 5: Core Logic and Proof Generation Functions
def clean_generated_text(text):
    text = str(text)
    text = re.sub(r'<\|.*?\|>$', '', text).strip() # Matches <|end|> or <|system|> etc. at string end
    text = re.sub(r'<\|[a-zA-Z]*$', '', text).strip() # Matches incomplete tokens like <|assistant at string end
    if st.session_state.tokenizer and st.session_state.tokenizer.eos_token:
        text = text.replace(st.session_state.tokenizer.eos_token, '')
    text = re.sub(r'\([\s\S]*?\)$', '', text).strip() # Removes (e.g. explanation) at the end
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
    text = re.sub(r'\n+', '\n', text).strip() # Normalize newlines
    if len(text) > 1 and text.startswith('"') and text.endswith('"'): text = text[1:-1]
    if len(text) > 1 and text.startswith("'") and text.endswith("'"): text = text[1:-1]
    text = re.sub(r"^[^\w\(]+", "", text) # Remove leading non-alphanumeric (keep opening parenthesis)
    return text.strip()

def generate_response_llm(prompt_instruction):
    if not st.session_state.model or not st.session_state.tokenizer:
        return "Error: Model or Tokenizer not available."
    if not st.session_state.tokenizer.pad_token_id or not st.session_state.tokenizer.eos_token_id:
        st.error("Tokenizer pad_token_id or eos_token_id is not set. This is critical.")
        return "Error: Tokenizer misconfiguration (pad/eos tokens)."

    system_message = "You are an AI assistant creating short, informal text message style excuses or apologies based on user requirements."
    full_prompt = f"<|system|>\n{system_message}<|end|>\n<|user|>\n{prompt_instruction}<|end|>\n<|assistant|>\n"
    
    inputs = st.session_state.tokenizer(full_prompt, return_tensors="pt", return_attention_mask=True)
    
    # Move inputs to model's device (GPU or CPU)
    model_device = st.session_state.model.device if hasattr(st.session_state.model, 'device') else 'cpu'
    inputs = {k: v.to(model_device) for k, v in inputs.items()}


    response_text = "[No Response]"
    try:
        with torch.no_grad():
            outputs = st.session_state.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=st.session_state.tokenizer.eos_token_id,
                pad_token_id=st.session_state.tokenizer.pad_token_id,
                num_return_sequences=1,
                use_cache=False # Important for Phi-3 stability with some setups
            )
        input_length = inputs['input_ids'].shape[1]
        generated_ids = outputs[0][input_length:]
        response_text = st.session_state.tokenizer.decode(generated_ids, skip_special_tokens=False) # Keep special for initial clean
        cleaned_response = clean_generated_text(response_text)
        if not cleaned_response: return "[Empty Response]"
        return cleaned_response
    except Exception as e:
        st.error(f"Generation Error: {e}")
        # traceback.print_exc() # Uncomment for detailed console error
        return f"Error: Generation failed ({type(e).__name__})"

def generate_voice_output(text, filename=GENERATED_AUDIO_PATH):
    if not text or text.startswith("Error:") or text == "[Empty Response]":
        st.warning("Skipping TTS for invalid or empty text.")
        return None
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(filename)
        return filename
    except Exception as e:
        st.error(f"Text-to-Speech (gTTS) Error: {e}")
        return None

def get_wrapped_text_and_size(draw, text, font, max_width):
    try:
        # Pillow's textbbox or multiline_textbbox is more reliable than heuristics
        # For multiline_textbbox, we need the text already wrapped.
        # textwrap.wrap is good for this.
        avg_char_width = getattr(font, 'size', DEFAULT_FONT_SIZE_MSG) * 0.55 # Heuristic for wrap width
        wrap_width = int(max_width / avg_char_width) if avg_char_width > 0 else 30
        
        lines = textwrap.wrap(str(text), width=wrap_width, replace_whitespace=True, drop_whitespace=True, break_long_words=True)
        wrapped_text = "\n".join(lines)
        
        # Get actual bounding box of the multiline text
        # x0, y0, x1, y1
        bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, spacing=4) # Pillow's method
        return wrapped_text, (bbox[2] - bbox[0], bbox[3] - bbox[1]) # width, height
    except Exception as e:
        st.warning(f"Text wrapping/sizing failed: {e}. Using basic split.")
        # Fallback
        lines = str(text).split('\n') 
        wrapped_text = "\n".join(lines)
        # Crude estimation for fallback
        height_est = len(lines) * getattr(font, 'size', DEFAULT_FONT_SIZE_MSG) * 1.5 
        return wrapped_text, (max_width, height_est)


def generate_whatsapp_screenshot(user_text, reply_text, filename=GENERATED_WHATSAPP_PATH):
    font_msg, font_info = None, None
    font_loaded_successfully = False

    if not st.session_state.font_available:
        st.warning(f"Custom font '{FONT_PATH}' not found. WhatsApp screenshot may look basic or fail.")
    else:
        try:
            font_msg = ImageFont.truetype(FONT_PATH, DEFAULT_FONT_SIZE_MSG)
            font_info = ImageFont.truetype(FONT_PATH, DEFAULT_FONT_SIZE_INFO)
            font_loaded_successfully = True
        except IOError as e:
            st.error(f"Error loading custom font '{FONT_PATH}': {e}. Will attempt default PIL font.")
    
    if not font_loaded_successfully: # Fallback if custom font failed or wasn't available
        try:
            font_msg = ImageFont.load_default() # Pillow's default bitmap font
            font_info = ImageFont.load_default() 
            st.info("Using default PIL font for WhatsApp screenshot.")
        except Exception as e:
            st.error(f"FATAL: Cannot load default PIL font: {e}. Screenshot generation aborted.")
            return None
    
    try:
        padding = 10; bubble_padding_h = 10; bubble_padding_v = 6; bubble_radius = 7
        img_width = 360; max_bubble_width_ratio = 0.78; inter_bubble_space = 8
        timestamp_space_in_bubble = 18; line_spacing = 4
        bg_color = (230, 221, 212); user_bubble_color = (213, 245, 190)
        reply_bubble_color = (255, 255, 255); text_color = (0, 0, 0)
        timestamp_color = (100, 111, 115); checkmark_color = (82, 178, 223)

        max_text_width_calc = (img_width - 2 * padding) * max_bubble_width_ratio
        
        dummy_img = Image.new('RGB', (1, 1)) # For text size calculation
        draw_dummy = ImageDraw.Draw(dummy_img)

        user_wrapped, user_text_size = get_wrapped_text_and_size(draw_dummy, user_text, font_msg, max_text_width_calc)
        reply_wrapped, reply_text_size = get_wrapped_text_and_size(draw_dummy, reply_text, font_msg, max_text_width_calc)
        
        # Ensure minimum sensible sizes
        user_text_w, user_text_h = max(user_text_size[0], 30), max(user_text_size[1], 15)
        reply_text_w, reply_text_h = max(reply_text_size[0], 30), max(reply_text_size[1], 15)


        user_bubble_w = max(min(user_text_w + 2 * bubble_padding_h + 65, img_width - 2*padding), 100) # +65 for timestamp area
        user_bubble_h = user_text_h + 2 * bubble_padding_v + timestamp_space_in_bubble
        reply_bubble_w = max(min(reply_text_w + 2 * bubble_padding_h, img_width - 2*padding), 50)
        reply_bubble_h = reply_text_h + 2 * bubble_padding_v

        img_height = padding + user_bubble_h + inter_bubble_space + reply_bubble_h + padding
        img = Image.new('RGB', (int(img_width), int(img_height)), color=bg_color)
        draw = ImageDraw.Draw(img)

        # Draw user message bubble (right-aligned)
        user_bubble_x = img_width - padding - user_bubble_w
        user_bubble_y = padding
        draw.rounded_rectangle((user_bubble_x, user_bubble_y, user_bubble_x + user_bubble_w, user_bubble_y + user_bubble_h), radius=bubble_radius, fill=user_bubble_color)
        draw.multiline_text((user_bubble_x + bubble_padding_h, user_bubble_y + bubble_padding_v), user_wrapped, font=font_msg, fill=text_color, spacing=line_spacing, align="left")

        timestamp_text = datetime.now().strftime("%H:%M")
        checkmarks = " ✓✓"
        try: # More precise placement with textlength (TrueType fonts)
             ts_bbox = draw.textbbox((0,0), timestamp_text, font=font_info)
             cm_bbox = draw.textbbox((0,0), checkmarks, font=font_info)
             ts_w, ts_h = ts_bbox[2]-ts_bbox[0], ts_bbox[3]-ts_bbox[1]
             cm_w, _ = cm_bbox[2]-cm_bbox[0], cm_bbox[3]-cm_bbox[1]

             total_len_ts_cm = ts_w + cm_w + 2 # 2 for spacing
             ts_cm_height = max(ts_h, 10) # Ensure some minimum height

             ts_x = user_bubble_x + user_bubble_w - bubble_padding_h - total_len_ts_cm
             cm_x = ts_x + ts_w + 2
             # Align to bottom of bubble, considering the text height
             ts_cm_y = user_bubble_y + user_bubble_h - bubble_padding_v - ts_cm_height 
             draw.text((ts_x, ts_cm_y), timestamp_text, font=font_info, fill=timestamp_color)
             draw.text((cm_x, ts_cm_y), checkmarks, font=font_info, fill=checkmark_color)
        except (AttributeError, TypeError): # Fallback for default PIL font or issues with textbbox
            ts_cm_combined = f"{timestamp_text}{checkmarks}"
            # Estimate width of combined text for fallback
            fallback_ts_cm_w = len(ts_cm_combined) * DEFAULT_FONT_SIZE_INFO * 0.5 
            fallback_ts_x = user_bubble_x + user_bubble_w - bubble_padding_h - fallback_ts_cm_w - 5 # 5 for margin
            fallback_ts_y = user_bubble_y + user_bubble_h - bubble_padding_v - DEFAULT_FONT_SIZE_INFO - 2
            draw.text((fallback_ts_x, fallback_ts_y), ts_cm_combined, font=font_info, fill=timestamp_color)


        # Draw reply message bubble (left-aligned)
        reply_bubble_x = padding
        reply_bubble_y = user_bubble_y + user_bubble_h + inter_bubble_space
        draw.rounded_rectangle((reply_bubble_x, reply_bubble_y, reply_bubble_x + reply_bubble_w, reply_bubble_y + reply_bubble_h), radius=bubble_radius, fill=reply_bubble_color)
        draw.multiline_text((reply_bubble_x + bubble_padding_h, reply_bubble_y + bubble_padding_v), reply_wrapped, font=font_msg, fill=text_color, spacing=line_spacing, align="left")

        img.save(filename)
        return filename
    except Exception as e:
        st.error(f"Error generating WhatsApp screenshot: {e}")
        # traceback.print_exc() # Uncomment for detailed console error
        return None

def generate_location_context(base_text, user_context=""):
    location_name, location_msg = "Unknown Location", "Delayed."
    if not st.session_state.model or not st.session_state.tokenizer: return "Error: Model/Tokenizer not loaded."

    context_hint = f"Context for excuse/apology: '{user_context}'. Original message: '{base_text}'." if user_context else f"Original message: '{base_text}'."
    prompt = (f"{context_hint}\nBased on this, generate a plausible location NAME (e.g., 'near Mill Road', 'at City Clinic') "
              f"AND a very short status message (max 10 words) explaining presence or delay related to the original message.\n"
              f"Provide the output strictly in this format:\nLocation: [Generated Location Name]\nMessage: [Generated Status Message]\nONLY these two lines.")

    raw_output = generate_response_llm(prompt)
    if raw_output.startswith("Error:"): return raw_output

    try:
        loc_match = re.search(r"Location:\s*(.*)", raw_output, re.IGNORECASE | re.MULTILINE)
        msg_match = re.search(r"Message:\s*(.*)", raw_output, re.IGNORECASE | re.MULTILINE)
        if loc_match: location_name = loc_match.group(1).strip() or location_name
        if msg_match: location_msg = msg_match.group(1).strip() or location_msg
    except Exception as e:
        st.warning(f"Location context parsing failed: {e}. Raw output: '{raw_output}'")

    return f"📍 Location: {location_name}\n💬 Status: {location_msg}"

def trigger_fake_emergency():
    caller_id, urgent_msg = "Unknown Caller", "Urgent - Call back ASAP."
    if not st.session_state.model or not st.session_state.tokenizer: return "Error: Model/Tokenizer not loaded."

    prompt = ("Generate a plausible caller ID for an urgent missed call (e.g., 'Mom', 'Work Emergency', 'Dr. Smith Office') "
              f"AND a short, urgent text message (max 15 words) left by them.\n"
              f"Provide the output strictly in this format:\nCaller: [Generated Caller ID]\nMessage: [Generated Urgent Message]\nONLY these two lines.")

    raw_output = generate_response_llm(prompt)
    if raw_output.startswith("Error:"): return f"Emergency generation failed: {raw_output}"

    try:
        caller_match = re.search(r"Caller:\s*(.*)", raw_output, re.IGNORECASE | re.MULTILINE)
        msg_match = re.search(r"Message:\s*(.*)", raw_output, re.IGNORECASE | re.MULTILINE)
        if caller_match: caller_id = caller_match.group(1).strip() or caller_id
        if msg_match: urgent_msg = msg_match.group(1).strip() or urgent_msg
    except Exception as e:
        st.warning(f"Fake emergency parsing failed: {e}. Raw output: '{raw_output}'")

    current_time = datetime.now().strftime("%I:%M %p")
    return (f"🚨 SIMULATED URGENT NOTIFICATION 🚨\n"
            f"📞 Missed Call From: {caller_id}\n"
            f"💬 Message: \"{urgent_msg}\"\n"
            f"🕒 Time: {current_time}")

def add_to_history(situation, priority, plausibility, msg_type, context, text, rating=None, favorite=False):
    new_entry_data = {
        'timestamp': datetime.now(), 'situation': situation, 'priority': priority,
        'plausibility': plausibility, 'message_type': msg_type, 'user_context': context,
        'generated_text': text, 'effectiveness_rating': rating, 'is_favorite': bool(favorite)
    }
    # Ensure all columns for the new entry are present, matching history_df schema
    for col in st.session_state.history_df.columns:
        if col not in new_entry_data: new_entry_data[col] = None # Or appropriate default
    
    new_entry_df = pd.DataFrame([new_entry_data], columns=st.session_state.history_df.columns) # Ensure column order
    try:
        st.session_state.history_df = pd.concat([st.session_state.history_df, new_entry_df], ignore_index=True)
    except Exception as e:
        st.error(f"Error adding to history: {e}")

def save_history_df():
    if isinstance(st.session_state.history_df, pd.DataFrame) and not st.session_state.history_df.empty:
        try:
            history_df_to_save = st.session_state.history_df.copy()
            history_df_to_save['is_favorite'] = history_df_to_save['is_favorite'].astype(bool)
            history_df_to_save['effectiveness_rating'] = pd.to_numeric(history_df_to_save['effectiveness_rating'], errors='coerce')
            history_df_to_save.to_csv(HISTORY_CSV_PATH, index=False)
            st.toast(f"History saved to '{HISTORY_CSV_PATH}'")
        except Exception as e:
            st.error(f"Error saving history: {e}")
    elif isinstance(st.session_state.history_df, pd.DataFrame) and st.session_state.history_df.empty:
        st.info("History is empty, nothing to save.")

def toggle_favorite_history(idx_str):
    try:
        idx = int(idx_str)
        if not (0 <= idx < len(st.session_state.history_df)):
            st.error(f"Error: Index {idx} is out of bounds.")
            return
        current_status = bool(st.session_state.history_df.loc[idx, 'is_favorite'])
        st.session_state.history_df.loc[idx, 'is_favorite'] = not current_status
        st.toast(f"Item ID {idx} {'marked as favorite' if not current_status else 'unmarked as favorite'}.")
        save_history_df()
    except ValueError:
        st.error(f"Error: Invalid index '{idx_str}'.")
    except KeyError: # If index somehow becomes invalid after check (less likely with direct .loc)
        st.error(f"Error: History item with index {idx} not found during toggle.")


def record_feedback_history(idx_str, rating_str):
    try:
        idx = int(idx_str)
        rating_val = int(rating_str)
        if not (0 <= idx < len(st.session_state.history_df)):
            st.error(f"Error: Index {idx} is out of bounds.")
            return False
        if not (0 <= rating_val <= 10):
            st.error(f"Error: Rating '{rating_val}' invalid. Must be 0-10.")
            return False
        st.session_state.history_df.loc[idx, 'effectiveness_rating'] = rating_val
        st.toast(f"Rating ({rating_val}/10) recorded for item ID {idx}.")
        save_history_df()
        return True
    except ValueError:
        st.error(f"Error: Invalid index or rating. Both must be integers.")
        return False
    except KeyError:
        st.error(f"Error: History item with index {idx} not found during feedback.")
        return False

def get_ranked_suggestions_from_history(situation, priority, plausibility, msg_type, context="", top_n=3):
    if not isinstance(st.session_state.history_df, pd.DataFrame) or st.session_state.history_df.empty or st.session_state.history_df['effectiveness_rating'].isna().all():
        return []

    hist_copy = st.session_state.history_df.copy()
    hist_copy['effectiveness_rating'] = pd.to_numeric(hist_copy['effectiveness_rating'], errors='coerce')
    hist_copy['timestamp'] = pd.to_datetime(hist_copy['timestamp'], errors='coerce')

    # Case-insensitive matching for string fields
    filtered_hist = hist_copy[
        (hist_copy['situation'].astype(str).str.lower() == str(situation).lower()) &
        (hist_copy['priority'].astype(str).str.lower() == str(priority).lower()) &
        (hist_copy['plausibility'].astype(str).str.lower() == str(plausibility).lower()) &
        (hist_copy['message_type'].astype(str).str.lower() == str(msg_type).lower()) &
        (hist_copy['effectiveness_rating'].notna()) # Only consider rated entries
    ].copy()

    if filtered_hist.empty:
        return []

    ranked_suggestions = filtered_hist.sort_values(by=['effectiveness_rating', 'timestamp'], ascending=[False, False])
    top_suggestions = []
    seen_texts = set()
    for _, row in ranked_suggestions.iterrows():
        if row['generated_text'] not in seen_texts:
            top_suggestions.append({'text': row['generated_text'], 'rating': row['effectiveness_rating']})
            seen_texts.add(row['generated_text'])
        if len(top_suggestions) >= top_n:
            break
    return top_suggestions

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Excuse/Apology Generator")
st.title("🎭 Intelligent Excuse and Apology Generator")
st.markdown(f"Using `{BASE_MODEL_ID}`")

# --- Initialization Section (runs once or when dependencies change) ---
if not st.session_state.app_initialized:
    with st.spinner("Initializing application... This may take a while for the first run."):
        st.info(f"GPU Available: {st.session_state.gpu_available}")
        if not st.session_state.gpu_available:
            st.warning("No GPU detected by PyTorch. Model loading will be on CPU (VERY SLOW).")

        if not st.session_state.font_available:
             st.warning(f"Font '{UPLOADED_FONT_NAME}' not found. WhatsApp proof might be basic.")
        else:
             st.success(f"Font '{UPLOADED_FONT_NAME}' found.")

        # Hugging Face Login - CORRECTED SECTION
        hf_token_from_secret = None
        final_hf_token = None 

        try:
            hf_token_from_secret = st.secrets.get("HUGGINGFACE_TOKEN")
            if hf_token_from_secret:
                st.sidebar.info("Hugging Face token loaded from secrets.")
                final_hf_token = hf_token_from_secret
        except (st.errors.StreamlitAPIException, FileNotFoundError): 
            st.sidebar.info("No secrets.toml file found or HUGGINGFACE_TOKEN not in secrets. Token can be entered manually if needed.")
        except Exception as e: 
            st.sidebar.warning(f"An error occurred while accessing secrets: {str(e)[:100]}")

        if not final_hf_token: # If token wasn't found in secrets
            hf_token_manual_input = st.sidebar.text_input(
                "Enter Hugging Face Token (Optional):",
                type="password",
                key="hf_token_manual_input_sidebar_init",
                help="Needed for private models or to avoid rate limits."
            )
            if hf_token_manual_input:
                final_hf_token = hf_token_manual_input
        
        if final_hf_token:
            try:
                hf_login_cli(token=final_hf_token, add_to_git_credential=False)
                st.sidebar.success("Logged into Hugging Face Hub.")
            except Exception as e:
                st.sidebar.error(f"Hugging Face login failed: {e}")
        else:
            st.sidebar.info("Proceeding without Hugging Face login (no token provided). Public models should still work.")

        # Load datasets
        _ = load_datasets() 

        # Load history
        load_history_df()

        # Load model
        is_adapter_present = os.path.isdir(FINE_TUNED_ADAPTER_DIR)
        if is_adapter_present:
            st.write(f"Found existing fine-tuned adapter: '{FINE_TUNED_ADAPTER_DIR}'.")
        
        st.session_state.model, st.session_state.tokenizer = load_model_and_tokenizer(
            is_fine_tuned_adapter_present=is_adapter_present, 
            load_on_cpu_flag=st.session_state.load_on_cpu
        )

        if st.session_state.model and st.session_state.tokenizer:
            st.success("Model and Tokenizer loaded successfully!")
            st.session_state.app_initialized = True
        else:
            st.error("Model and/or Tokenizer failed to load. The application may not function correctly. Check messages above.")
            # No st.stop() here, main app flow will check st.session_state.app_initialized

# --- Main Application Flow ---
if not st.session_state.app_initialized:
    st.error("Application initialization failed. Model and/or Tokenizer are not available. Please check the error messages above and ensure all prerequisites are met (e.g., file paths, Hugging Face login if needed). You might need to restart the app after fixing issues.")
    st.stop() # Stop execution if app didn't initialize properly

# --- Main Application UI (Tabs etc.) ---
tab1, tab2, tab3 = st.tabs(["🆕 Generate New", "📚 History", "🆘 Other Proofs"])

with tab1:
    st.header("Craft Your Message")
    with st.form("generation_form"):
        col1, col2 = st.columns(2)
        with col1:
            situation = st.text_input("Situation (e.g., 'late for meeting', 'missed deadline')", key="sit", help="Describe the general scenario.").strip().lower()
            priority = st.selectbox("Priority", ["low", "medium", "high"], index=1, key="pri", help="How urgent should the message sound?").strip().lower()
        with col2:
            plausibility = st.selectbox("Plausibility", ["low", "medium", "high"], index=1, key="pla", help="How believable should the message be?").strip().lower()
            msg_type = st.selectbox("Message Type", ["excuse", "apology"], key="mty", help="Is it an excuse or an apology?").strip().lower()
        
        context = st.text_area("Additional Context/Reason (be specific for better results)", key="con", help="Provide specific details for the AI to use.").strip()
        
        submit_button = st.form_submit_button("✨ Generate Message")

    if submit_button:
        if not situation or not context:
            st.error("'Situation' and 'Context/Reason' cannot be empty.")
        else:
            with st.spinner("Thinking of the perfect words..."):
                # Show past suggestions
                st.subheader("💡 Previously Rated Suggestions")
                ranked_suggestions = get_ranked_suggestions_from_history(situation, priority, plausibility, msg_type, context)
                if ranked_suggestions:
                    for i, sug in enumerate(ranked_suggestions):
                        st.markdown(f"  {i+1}. (Rated: {sug['rating']:.0f}/10) \"{sug['text']}\"")
                else:
                    st.info("No similar rated examples found in history.")
                st.divider()

                # Generate new message
                st.subheader(f"💬 Generated {msg_type.capitalize()}")
                generation_prompt = (
                    f"Generate a short, informal text message style {msg_type} for the situation: '{situation}'. "
                    f"The specific reason or context is: '{context}'. "
                    f"The message should sound like it has {plausibility} plausibility and {priority} urgency. "
                    f"Keep it concise (1-2 sentences typically). Avoid formal greetings or sign-offs. "
                    f"Output ONLY the message text itself."
                )
                generated_text = generate_response_llm(generation_prompt)
                
                if not generated_text or generated_text.startswith("Error:") or generated_text == "[Empty Response]":
                    st.error(f"Message generation failed: {generated_text}")
                else:
                    st.session_state.current_generated_text = generated_text
                    st.markdown(f"#### \"{generated_text}\"")
                    
                    add_to_history(situation, priority, plausibility, msg_type, context, generated_text)
                    st.session_state.current_history_index = len(st.session_state.history_df) - 1
                    save_history_df()
                    st.success(f"Message (ID: {st.session_state.current_history_index}) added to history.")

    if st.session_state.current_generated_text and st.session_state.current_history_index is not None and \
       0 <= st.session_state.current_history_index < len(st.session_state.history_df): # Ensure index is valid
        st.divider()
        st.subheader("🌟 Feedback & Proofs for Current Message")
        st.markdown(f"**Message (ID {st.session_state.current_history_index}):** \"{st.session_state.current_generated_text}\"")

        # Feedback
        fb_col1, fb_col2 = st.columns([3,1]) # fb_col2 not used, but keeps structure if needed later
        
        current_rating = st.session_state.history_df.loc[st.session_state.current_history_index, 'effectiveness_rating']
        current_rating = int(current_rating) if pd.notna(current_rating) else None

        new_rating = st.number_input(f"Rate effectiveness (0-10) for ID {st.session_state.current_history_index}:", 
                                     min_value=0, max_value=10, step=1, 
                                     value=current_rating, 
                                     key=f"rating_{st.session_state.current_history_index}")
        
        if new_rating is not None and new_rating != current_rating : # Check if user interacted and value changed
            if record_feedback_history(str(st.session_state.current_history_index), str(new_rating)):
                # st.success(f"Rating {new_rating}/10 saved for ID {st.session_state.current_history_index}.") # Toast is enough
                st.rerun() 

        is_fav = bool(st.session_state.history_df.loc[st.session_state.current_history_index, 'is_favorite'])
        fav_button_text = "❤️ Unfavorite" if is_fav else "🤍 Favorite"
        if st.button(fav_button_text, key=f"fav_btn_{st.session_state.current_history_index}"):
            toggle_favorite_history(str(st.session_state.current_history_index))
            st.rerun()

        # Proofs for current message
        st.markdown("---")
        st.write("**Proofs:**")
        p_col1, p_col2, p_col3 = st.columns(3) # Adjusted for three buttons
        with p_col1:
            if st.button("🎤 Generate Voice Note", key="tts_btn", use_container_width=True):
                with st.spinner("Generating audio..."):
                    audio_path = generate_voice_output(st.session_state.current_generated_text)
                    if audio_path:
                        st.audio(audio_path)
                    # Error handled in function

        with p_col2:
            if st.button("📱 Simulate WhatsApp", key="wa_btn", use_container_width=True):
                with st.spinner("Generating WhatsApp screenshot..."):
                    whatsapp_img_path = generate_whatsapp_screenshot(st.session_state.current_generated_text, random.choice(RECIPIENT_REPLIES))
                    if whatsapp_img_path:
                        st.image(whatsapp_img_path, width=360)
                    # Error handled in function
        
        with p_col3:
            if st.button("📍 Generate Location Context", key="loc_btn", use_container_width=True):
                with st.spinner("Determining location context..."):
                    loc_context_user_context = st.session_state.history_df.loc[st.session_state.current_history_index, 'user_context']
                    location_output = generate_location_context(st.session_state.current_generated_text, user_context=loc_context_user_context)
                    st.text_area("Generated Location Context:", value=location_output, height=100, disabled=True, key="loc_out_disp")


with tab2:
    st.header("📜 Generation History")
    
    if st.session_state.history_df.empty:
        st.info("No history records yet.")
    else:
        show_fav_only = st.checkbox("Show Favorites Only", key="fav_filter_hist_tab")
        
        df_display_hist = st.session_state.history_df.copy()
        df_display_hist['is_favorite'] = df_display_hist['is_favorite'].astype(bool)
        df_display_hist['effectiveness_rating'] = pd.to_numeric(df_display_hist['effectiveness_rating'], errors='coerce').astype('Int64')

        if show_fav_only:
            df_display_hist = df_display_hist[df_display_hist['is_favorite']].copy()

        if df_display_hist.empty:
            st.info(f"No {'favorites' if show_fav_only else 'history records'} found matching criteria.")
        else:
            display_cols_hist = ['ID','timestamp', 'situation', 'message_type', 'user_context', 'generated_text', 'effectiveness_rating', 'is_favorite']
            df_view_hist = df_display_hist.reset_index().rename(columns={'index': 'ID'})
            df_view_hist = df_view_hist[[col for col in display_cols_hist if col in df_view_hist.columns]]
            df_view_hist['timestamp'] = pd.to_datetime(df_view_hist['timestamp']).dt.strftime('%Y-%m-%d %H:%M') # Ensure it's datetime before strftime
            df_view_hist['effectiveness_rating'] = df_view_hist['effectiveness_rating'].apply(lambda x: f"{x}/10" if pd.notna(x) else "N/A")
            
            st.dataframe(df_view_hist, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("Manage History Item")
             # Use df_view_hist which is already filtered and has 'ID'
            if not df_view_hist.empty:
                # Provide a selection of IDs from the currently displayed table
                available_ids = df_view_hist['ID'].tolist()
                selected_id_manage_hist = st.selectbox(f"Select ID to manage:", options=available_ids, index=None, key="manage_id_hist_tab", placeholder="Choose an ID...")
                
                if selected_id_manage_hist is not None: # Check if user selected an ID
                    try:
                        selected_id = int(selected_id_manage_hist) # Already an int from selectbox if options are ints
                        # We use the selected_id directly as it comes from the displayed (potentially filtered) dataframe's 'ID' column
                        # which corresponds to the original index in st.session_state.history_df
                        if 0 <= selected_id < len(st.session_state.history_df): 
                            st.write(f"**Managing Item ID {selected_id}:** \"{st.session_state.history_df.loc[selected_id, 'generated_text']}\"")
                            
                            hist_rating_val = st.session_state.history_df.loc[selected_id, 'effectiveness_rating']
                            hist_rating_val = int(hist_rating_val) if pd.notna(hist_rating_val) else None

                            new_rating_hist_item = st.number_input(f"New rating for ID {selected_id} (0-10):", 
                                                                min_value=0, max_value=10, step=1, value=hist_rating_val,
                                                                key=f"hist_rate_item_{selected_id}")
                            if new_rating_hist_item is not None and new_rating_hist_item != hist_rating_val:
                                 if record_feedback_history(str(selected_id), str(new_rating_hist_item)):
                                    st.rerun()

                            is_fav_hist_item = bool(st.session_state.history_df.loc[selected_id, 'is_favorite'])
                            fav_button_hist_item_text = "❤️ Unfavorite" if is_fav_hist_item else "🤍 Favorite"
                            if st.button(fav_button_hist_item_text, key=f"hist_fav_btn_item_{selected_id}"):
                                toggle_favorite_history(str(selected_id))
                                st.rerun()
                        else: # Should not happen if selected_id comes from available_ids based on df_view_hist
                            st.warning(f"Selected ID {selected_id} is somehow out of range of the main history.")
                    except ValueError: # Should not happen with selectbox of ints
                        st.warning("Invalid ID selected.")
            else: # If df_view_hist (potentially filtered) is empty
                st.info("No items to manage in the current view.")


with tab3:
    st.header("🚨 Other Simulated Proofs")
    if st.button("💥 Trigger Fake Emergency Notification", key="emergency_btn_tab3", use_container_width=True):
        with st.spinner("Simulating emergency..."):
            emergency_output = trigger_fake_emergency()
            st.text_area("Simulated Emergency:", value=emergency_output, height=150, disabled=True, key="emergency_disp_tab3")
            st.balloons()


st.sidebar.markdown("---")
st.sidebar.header("⚠️ Status & Controls")
if st.session_state.gpu_available:
    st.sidebar.success("GPU Detected by PyTorch.")
else:
    st.sidebar.warning("No GPU Detected by PyTorch. CPU operation will be slow.")

if st.session_state.font_available:
    st.sidebar.success(f"Font '{UPLOADED_FONT_NAME}' loaded.")
else:
    st.sidebar.error(f"Font '{UPLOADED_FONT_NAME}' not found.")

if st.session_state.model and st.session_state.tokenizer:
    st.sidebar.success("LLM Model & Tokenizer are loaded.")
    if st.session_state.is_model_fine_tuned:
        st.sidebar.info("Fine-tuned adapter is active.")
    else:
        st.sidebar.info("Base model is active.")
else:
    st.sidebar.error("LLM Model & Tokenizer NOT loaded.")

if st.sidebar.button("🔁 Save History Manually", key="save_hist_manual_btn"):
    save_history_df()

st.sidebar.markdown("---")
st.sidebar.caption(f"Base Model: {BASE_MODEL_ID}")
st.sidebar.caption(f"History File: {HISTORY_CSV_PATH}")