# excuse_generator_logic.py
import torch
import pandas as pd
import numpy as np
import random
import os
import json
from datetime import datetime
import textwrap
import re
import traceback
import tempfile

# Hugging Face Libraries
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    logging as hf_logging
)
from peft import PeftModel
from huggingface_hub import login as hf_login

# Proof Generation & Utilities
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont

# Suppress excessive Hugging Face warnings (Changed from warning to error for less verbosity)
hf_logging.set_verbosity_error()

# --- Constants ---
BASE_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
FINE_TUNED_ADAPTER_DIR = "phi3-mini-excuse-generator-adapter" # Relative path or absolute
RECIPIENT_REPLIES = ["Okay", "Alright, thanks", "Got it.", "Ok, hope you're alright?", "Understood.", "👍", "Sure thing.", "Ok.", "No worries.", "Seen."]
DEFAULT_FONT_SIZE_MSG = 16
DEFAULT_FONT_SIZE_INFO = 11
HISTORY_DF_COLUMNS = ['id', 'timestamp', 'situation', 'priority', 'plausibility', 'message_type', 'user_context', 'generated_text', 'effectiveness_rating', 'is_favorite']


def initialize_model_and_tokenizer(model_id=BASE_MODEL_ID, adapter_dir=FINE_TUNED_ADAPTER_DIR, hf_token=None):
    """Loads the language model and tokenizer, attempting GPU (CUDA or MPS) first."""
    model = None
    tokenizer = None
    is_model_fine_tuned = False
    actual_device = "cpu" # Default to CPU

    if hf_token:
        try:
            hf_login(token=hf_token)
            print("Hugging Face login successful.")
        except Exception as e:
            print(f"Hugging Face login failed: {e}")

    try:
        # 1. Check for NVIDIA CUDA GPU
        if torch.cuda.is_available():
            actual_device = "cuda"
            print(f"NVIDIA GPU detected. Attempting to load model on {actual_device}.")
            # Configure BitsAndBytes for 4-bit quantization on CUDA
            bnb_config_inference = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config_inference,
                device_map="auto",  # Handles multi-GPU or fitting large models on CUDA
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 # Compute dtype for dequantized weights
            )
            print(f"Base model '{model_id}' loaded on GPU (quantized).")

        # 2. Else, check for Apple Silicon MPS GPU
        elif torch.backends.mps.is_available():
            actual_device = "mps"
            print(f"Apple Silicon GPU (MPS) detected. Attempting to load model on {actual_device}.")
            # BitsAndBytes 4-bit quantization is NOT available for MPS.
            # Load in a compatible dtype, e.g., bfloat16. Phi-3 supports bfloat16.
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 # Use bfloat16 for Phi-3 on MPS
            )
            model.to(actual_device) # Move the model to the MPS device
            print(f"Base model '{model_id}' loaded on MPS device with torch_dtype=torch.bfloat16.")

        # 3. Else, fallback to CPU
        else:
            print("No CUDA or MPS GPU detected. Attempting to load model on CPU (will be slow).")
            actual_device = "cpu"
            model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
            # model will be on CPU by default if loaded this way
            print(f"Base model '{model_id}' loaded on CPU.")


        # Attempt to load fine-tuned adapter if directory exists
        if adapter_dir and os.path.isdir(adapter_dir) and model: # Check if model loaded successfully
            print(f"Attempting to load fine-tuned adapter from '{adapter_dir}'...")
            try:
                # PeftModel will be loaded onto the same device as the base model.
                model = PeftModel.from_pretrained(model, adapter_dir)
                is_model_fine_tuned = True
                # Ensure model (with adapter) is on the target device, especially for MPS.
                model.to(actual_device)
                print(f"Fine-tuned PEFT adapter loaded successfully. Model is now on device: {model.device.type}")
            except Exception as e:
                print(f"WARNING: Failed to load PEFT adapter: {e}. Using base model only.")
                traceback.print_exc()
                is_model_fine_tuned = False
        elif model: # Model loaded, but no adapter
            print("No fine-tuned adapter directory specified or found. Using base model.")
        elif not model: # Model failed to load
             print("Base model failed to load. Cannot load adapter.")


        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                print("Set tokenizer pad_token to eos_token.")
            else: # Fallback if no eos_token
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                if model: model.resize_token_embeddings(len(tokenizer)) # Resize model embeddings if new token added
                print("Added new PAD token as tokenizer had no pad_token or eos_token.")
        tokenizer.padding_side = "left" # Important for causal LMs (left padding for generation)
        print(f"Tokenizer for '{model_id}' loaded. Padding side: {tokenizer.padding_side}.")

    except Exception as e:
        print(f"FATAL ERROR during model/tokenizer initialization: {e}")
        traceback.print_exc()
        return None, None, False, "cpu" # Fallback to CPU on error

    # Final check on model's device
    if model:
        final_device_type = model.device.type
        print(f"Model final device type: {final_device_type}")
        # Update actual_device to reflect reality if there's a discrepancy
        if actual_device != final_device_type:
            if not (actual_device.startswith(final_device_type) or final_device_type.startswith(actual_device)): # e.g. cuda vs cuda:0
                 print(f"WARNING: Device mismatch or specification. Expected/Initial: {actual_device}, Final: {final_device_type}. Using final.")
            actual_device = final_device_type
    else:
        # Model is None, means loading failed.
        actual_device = "cpu" # Ensure it's CPU if model is None
        print("Model is None after initialization attempts, setting device to CPU.")

    return model, tokenizer, is_model_fine_tuned, actual_device


def load_dataframes_from_objects(excuse_file_obj, apology_file_obj, history_file_obj=None):
    """Loads datasets from uploaded file objects and history if provided."""
    combined_df = pd.DataFrame()
    history_df = pd.DataFrame(columns=HISTORY_DF_COLUMNS)
    history_id_counter = 0

    standard_cols_map = {'scenario': 'situation', 'urgency': 'priority', 'believability': 'plausibility', 'excuse': 'message_text', 'apology': 'message_text'}
    standard_cols_final = ['situation', 'priority', 'plausibility', 'message_text']
    excuse_original_cols = ['scenario', 'urgency', 'believability', 'excuse']
    apology_original_cols = ['scenario', 'urgency', 'believability', 'apology']

    try:
        if excuse_file_obj:
            excuse_df_raw = pd.read_csv(excuse_file_obj)
            missing_excuse_cols = [col for col in excuse_original_cols if col not in excuse_df_raw.columns]
            if missing_excuse_cols: raise KeyError(f"Excuse CSV missing: {missing_excuse_cols}")
            excuse_df_raw.dropna(subset=excuse_original_cols, inplace=True)
            excuse_df = excuse_df_raw[excuse_original_cols].copy()
            excuse_df.rename(columns=standard_cols_map, inplace=True)
            excuse_df['message_type'] = 'excuse'
            combined_df = pd.concat([combined_df, excuse_df[standard_cols_final + ['message_type']]], ignore_index=True)
            print(f"Loaded and processed {len(excuse_df)} excuses.")

        if apology_file_obj:
            apology_df_raw = pd.read_csv(apology_file_obj)
            missing_apology_cols = [col for col in apology_original_cols if col not in apology_df_raw.columns]
            if missing_apology_cols: raise KeyError(f"Apology CSV missing: {missing_apology_cols}")
            apology_df_raw.dropna(subset=apology_original_cols, inplace=True)
            apology_df = apology_df_raw[apology_original_cols].copy()
            apology_df.rename(columns=standard_cols_map, inplace=True)
            apology_df['message_type'] = 'apology'
            combined_df = pd.concat([combined_df, apology_df[standard_cols_final + ['message_type']]], ignore_index=True)
            print(f"Loaded and processed {len(apology_df)} apologies.")

        if not combined_df.empty:
            combined_df = combined_df.astype(str)
            print(f"Total combined entries for fine-tuning/context: {len(combined_df)}")

        # Load History
        if history_file_obj:
            history_df_loaded = pd.read_csv(history_file_obj, low_memory=False)
            # Ensure all expected columns are present
            for col in HISTORY_DF_COLUMNS:
                if col not in history_df_loaded.columns:
                    history_df_loaded[col] = None if col != 'id' else 0 # Default for ID
            history_df = history_df_loaded[HISTORY_DF_COLUMNS].copy() # Select and order
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'], errors='coerce')
            history_df['effectiveness_rating'] = pd.to_numeric(history_df['effectiveness_rating'], errors='coerce')
            history_df['is_favorite'] = history_df['is_favorite'].apply(lambda x: str(x).strip().lower() == 'true' if pd.notna(x) else False).astype(bool)
            history_df['id'] = pd.to_numeric(history_df['id'], errors='coerce').fillna(0).astype(int)
            history_df.dropna(subset=['timestamp'], inplace=True) # Drop rows where timestamp couldn't be parsed
            if not history_df.empty and 'id' in history_df.columns and history_df['id'].notna().any():
                history_id_counter = history_df['id'].max() + 1
            else:
                history_id_counter = 0
            print(f"Loaded {len(history_df)} history entries. Next ID: {history_id_counter}")
        else:
             print("No history file provided, starting fresh.")


    except Exception as e:
        print(f"Error loading dataframes: {e}")
        traceback.print_exc()
        # Return empty DataFrames on error but allow app to continue if possible
    return combined_df, history_df, history_id_counter


def clean_generated_text(text, tokenizer_for_eos=None):
    text = str(text)
    # More robustly remove special tokens like <|end|> or <|user|>
    text = re.sub(r'<\|.*?\|>', '', text).strip() # Remove any token like <|...|>
    text = re.sub(r'<\|[a-zA-Z_]*$', '', text).strip() # Remove partially formed tokens at the end

    if tokenizer_for_eos and tokenizer_for_eos.eos_token:
        text = text.replace(tokenizer_for_eos.eos_token, '')

    # Remove incomplete or standalone typical instruction/role markers if they appear malformed
    text = re.sub(r"(<\|system\|>|<\|user\|>|<\|assistant\|>)$", "", text.strip()).strip()

    text = re.sub(r'\([\s\S]*?\)$', '', text).strip() # Remove trailing parenthesized text (often model explanations)
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
    text = re.sub(r'\n+', '\n', text).strip() # Normalize newlines

    # Remove leading/trailing quotes if they encompass the whole string
    if len(text) > 1 and text.startswith('"') and text.endswith('"'): text = text[1:-1]
    if len(text) > 1 and text.startswith("'") and text.endswith("'"): text = text[1:-1]

    # A common pattern is "Excuse: Actual message" or "Apology: Actual message"
    # Try to strip these prefixes if the model adds them.
    text = re.sub(r"^(excuse|apology|message|output|text)\s*:\s*", "", text, flags=re.IGNORECASE).strip()

    # Remove leading non-alphanumeric characters that are not part of a typical message start
    # This is a bit gentler than the original r"^[^\w\(]+"
    text = re.sub(r"^(?:[^\w\"\'\(\ अरे भाई]\s*)+", "", text)


    return text.strip()

def generate_response(model, tokenizer, prompt_instruction, device):
    if not model or not tokenizer: return "Error: Model/Tokenizer not available."
    # Check for pad_token_id and eos_token_id existence on the tokenizer object itself
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        return "Error: Tokenizer pad_token_id or eos_token_id is missing. Please check tokenizer setup."

    system_message = "You are an AI assistant creating short, informal text message style excuses or apologies based on user requirements."
    full_prompt = f"<|system|>\n{system_message}<|end|>\n<|user|>\n{prompt_instruction}<|end|>\n<|assistant|>\n"

    try:
        inputs = tokenizer(full_prompt, return_tensors="pt", return_attention_mask=True)
        # Move inputs to the model's device (e.g., 'cuda', 'mps', or 'cpu')
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                num_return_sequences=1,
                use_cache=False # *** THIS IS THE KEY CHANGE TO FIX THE ERROR ***
            )
        input_length = inputs['input_ids'].shape[1]
        # Handle cases where output might be shorter than input (e.g., if max_new_tokens is very small)
        if outputs.shape[1] > input_length:
            generated_ids = outputs[0][input_length:]
            response_text = tokenizer.decode(generated_ids, skip_special_tokens=False) # Keep special tokens for cleaning
        else: # If no new tokens were generated
            response_text = ""

        cleaned_response = clean_generated_text(response_text, tokenizer)
        if not cleaned_response: return "[Empty Response]"
        return cleaned_response
    except Exception as e:
        traceback.print_exc()
        return f"Error: Generation failed ({type(e).__name__}: {e})"


def generate_voice_output_st(text, output_dir=None):
    if not text or text.startswith("Error:") or text == "[Empty Response]": return None
    try:
        # Ensure output_dir exists, create if not
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created temporary directory for TTS: {output_dir}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", dir=output_dir, mode="wb") as tmpfile:
            tts = gTTS(text=text, lang='en', slow=False)
            tts.write_to_fp(tmpfile) # Write to the file object
            return tmpfile.name # Return the path to the temp file
    except Exception as e:
        print(f"TTS Error: {e}")
        traceback.print_exc()
        return None

def get_wrapped_text_and_size(draw, text, font, max_width, default_font_size=DEFAULT_FONT_SIZE_MSG):
    try:
        font_size_to_use = default_font_size
        if hasattr(font, 'size'): # For TrueType fonts
            font_size_to_use = font.size
        elif hasattr(font, 'getmetrics'): # For default PIL fonts (approximate)
             # For default fonts, getmetrics() returns (width, height) for the entire font
             # This is a rough heuristic for average char width
            font_metrics = font.getmetrics()
            avg_char_width = font_metrics[0] if font_metrics else font_size_to_use * 0.6
        else: # Fallback
            avg_char_width = font_size_to_use * 0.6

        wrap_width = int(max_width / avg_char_width) if avg_char_width > 0 else 30
        wrap_width = max(10, wrap_width) # Ensure a minimum wrap width

        lines = textwrap.wrap(str(text), width=wrap_width, replace_whitespace=False, drop_whitespace=False)
        wrapped_text = "\n".join(lines)

        # Use textbbox for more accurate size calculation with multiline text
        bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, spacing=4)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        return wrapped_text, (text_width, text_height)

    except Exception as e: # Fallback
        print(f"Warning: Error in get_wrapped_text_and_size: {e}. Using fallback.")
        lines = str(text).split('\n')
        estimated_height = len(lines) * default_font_size * 1.5 # Rough estimate
        return "\n".join(lines), (max_width, estimated_height)


def generate_whatsapp_screenshot_st(user_text, reply_text, font_path_param,
                                    default_font_size_msg=DEFAULT_FONT_SIZE_MSG,
                                    default_font_size_info=DEFAULT_FONT_SIZE_INFO,
                                    output_dir=None):
    font_msg, font_info = None, None
    try:
        if font_path_param and os.path.exists(font_path_param):
            try:
                font_msg = ImageFont.truetype(font_path_param, default_font_size_msg)
                font_info = ImageFont.truetype(font_path_param, default_font_size_info)
            except IOError:
                print(f"Error loading custom font from {font_path_param}, using default.")
                font_msg = ImageFont.load_default()
                font_info = ImageFont.load_default()
        else:
            font_msg = ImageFont.load_default()
            font_info = ImageFont.load_default()
    except Exception as e:
        print(f"FATAL: Cannot load any font (custom or default): {e}. Screenshot generation will fail."); return None


    try:
        # Ensure output_dir exists
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created temporary directory for WhatsApp screenshot: {output_dir}")

        padding = 10; bubble_padding_h = 12; bubble_padding_v = 8; bubble_radius = 10 # Slightly increased padding/radius
        img_width = 360; max_bubble_width_ratio = 0.78; inter_bubble_space = 8
        timestamp_space_in_bubble = 20; line_spacing = 4 # Increased space for timestamp
        bg_color = (229, 221, 213); user_bubble_color = (218, 247, 191); reply_bubble_color = (255, 255, 255) # WhatsApp green/white
        text_color = (20, 20, 20); timestamp_color = (105, 120, 128); checkmark_color = (82, 178, 223) # Standard colors

        max_text_width_for_bubble = (img_width - 2 * padding - 2 * bubble_padding_h) * max_bubble_width_ratio

        # Dummy draw for text size calculation
        dummy_img = Image.new('RGB', (1, 1)); draw_dummy = ImageDraw.Draw(dummy_img)

        user_wrapped, user_text_size = get_wrapped_text_and_size(draw_dummy, user_text, font_msg, max_text_width_for_bubble, default_font_size_msg)
        reply_wrapped, reply_text_size = get_wrapped_text_and_size(draw_dummy, reply_text, font_msg, max_text_width_for_bubble, default_font_size_msg)

        # Calculate bubble dimensions based on text content
        user_bubble_content_w = user_text_size[0]
        user_bubble_content_h = user_text_size[1]
        # Estimate width for timestamp and checkmarks
        timestamp_text = datetime.now().strftime("%H:%M")
        checkmarks = " ✓✓"
        ts_cm_combined_text = timestamp_text + checkmarks
        # Use textlength if available and font_info is a TrueType font
        try:
            ts_cm_width = draw_dummy.textlength(ts_cm_combined_text, font=font_info)
        except AttributeError: # Fallback for default font or if textlength not present
             ts_cm_width = len(ts_cm_combined_text) * (default_font_size_info * 0.6) # Estimate

        # User bubble width: max of text width and timestamp width, plus padding
        user_bubble_w = max(user_bubble_content_w, ts_cm_width + 5) + 2 * bubble_padding_h # +5 for spacing
        user_bubble_w = min(user_bubble_w, img_width - 2 * padding) # Cap at image width
        user_bubble_h = user_bubble_content_h + 2 * bubble_padding_v + timestamp_space_in_bubble

        reply_bubble_content_w = reply_text_size[0]
        reply_bubble_content_h = reply_text_size[1]
        # Reply bubble width: text width plus padding
        reply_bubble_w = reply_bubble_content_w + 2 * bubble_padding_h
        reply_bubble_w = min(reply_bubble_w, img_width - 2 * padding) # Cap at image width
        reply_bubble_h = reply_bubble_content_h + 2 * bubble_padding_v # No extra timestamp space needed in reply bubble itself

        img_height = padding + user_bubble_h + inter_bubble_space + reply_bubble_h + padding
        img = Image.new('RGB', (int(img_width), int(img_height)), color=bg_color); draw = ImageDraw.Draw(img)

        # User message bubble (right-aligned)
        user_bubble_x = img_width - padding - user_bubble_w
        user_bubble_y = padding
        draw.rounded_rectangle((user_bubble_x, user_bubble_y, user_bubble_x + user_bubble_w, user_bubble_y + user_bubble_h), radius=bubble_radius, fill=user_bubble_color)
        draw.multiline_text((user_bubble_x + bubble_padding_h, user_bubble_y + bubble_padding_v), user_wrapped, font=font_msg, fill=text_color, spacing=line_spacing, align="left")

        # Timestamp and checkmarks for user message
        try:
            # Get height of a single line of info text for precise vertical alignment
            info_line_bbox = draw_dummy.textbbox((0,0), "12:34 ✓✓", font=font_info)
            info_line_height = info_line_bbox[3] - info_line_bbox[1]

            ts_cm_y = user_bubble_y + user_bubble_h - bubble_padding_v - info_line_height

            # Position timestamp and checkmarks from the right edge of the text area inside the bubble
            # Aligning timestamp to the right of the text content area
            ts_bbox = draw_dummy.textbbox((0,0), timestamp_text, font=font_info)
            ts_width = ts_bbox[2] - ts_bbox[0]
            cm_bbox = draw_dummy.textbbox((0,0), checkmarks, font=font_info)
            cm_width = cm_bbox[2] - cm_bbox[0]

            cm_x = user_bubble_x + user_bubble_w - bubble_padding_h - cm_width
            ts_x = cm_x - ts_width - 2 # 2px space between timestamp and checkmarks

            draw.text((ts_x, ts_cm_y), timestamp_text, font=font_info, fill=timestamp_color)
            draw.text((cm_x, ts_cm_y), checkmarks, font=font_info, fill=checkmark_color)
        except Exception as e_ts:
            print(f"Minor issue positioning timestamp/checkmarks: {e_ts}. Using fallback.")
            draw.text((user_bubble_x + user_bubble_w - bubble_padding_h - 65, user_bubble_y + user_bubble_h - bubble_padding_v - 15), f"{timestamp_text}{checkmarks}", font=font_info, fill=timestamp_color)


        # Reply message bubble (left-aligned)
        reply_bubble_x = padding
        reply_bubble_y = user_bubble_y + user_bubble_h + inter_bubble_space
        draw.rounded_rectangle((reply_bubble_x, reply_bubble_y, reply_bubble_x + reply_bubble_w, reply_bubble_y + reply_bubble_h), radius=bubble_radius, fill=reply_bubble_color)
        draw.multiline_text((reply_bubble_x + bubble_padding_h, reply_bubble_y + bubble_padding_v), reply_wrapped, font=font_msg, fill=text_color, spacing=line_spacing, align="left")
        # Optionally, add timestamp for reply too (typically WhatsApp shows it on hover or for the last message in sequence)
        # For simplicity, this example doesn't add a visible timestamp to the reply bubble directly.

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=output_dir, mode="wb") as tmpfile:
            img.save(tmpfile, format="PNG")
            return tmpfile.name
    except Exception as e:
        print(f"Error generating WhatsApp screenshot: {e}"); traceback.print_exc(); return None


def generate_location_context(model, tokenizer, base_text, user_context, device):
    if not model or not tokenizer: return "Error: Model/Tokenizer not loaded."
    location_name, location_msg = "Unknown Location", "Delayed."
    context_hint = f"Reason: '{user_context}'. Original message for which this location context is being generated: '{base_text}'." if user_context else f"Original message for which this location context is being generated: '{base_text}'."
    # The prompt for generate_response should be just the user instruction part
    user_prompt_instruction = (
        f"{context_hint}\nGenerate a plausible location name and a short status message.\n"
        f"Format:\nLocation: [Location Name]\nMessage: [Short status message]\nOutput ONLY these two lines."
    )
    raw_output = generate_response(model, tokenizer, user_prompt_instruction, device)

    if raw_output.startswith("Error:") or raw_output == "[Empty Response]":
        print(f"Location context generation failed or empty: {raw_output}")
        return f"📍 Loc: {location_name}\n💬 Status: {location_msg} (Generation Error)"

    try:
        loc_match = re.search(r"Location:\s*(.*)", raw_output, re.IGNORECASE | re.MULTILINE)
        msg_match = re.search(r"Message:\s*(.*)", raw_output, re.IGNORECASE | re.MULTILINE)

        if loc_match: location_name = loc_match.group(1).strip() or location_name
        if msg_match: location_msg = msg_match.group(1).strip() or location_msg

        # Clean up potential model verbosity if it didn't strictly follow format
        location_name = location_name.split('\n')[0]
        location_msg = location_msg.split('\n')[0]

    except Exception as e: print(f"WARN: Location context parsing failed: {e}. Raw output: '{raw_output}'")
    return f"📍 Loc: {location_name}\n💬 Status: {location_msg}"

def trigger_fake_emergency(model, tokenizer, device):
    if not model or not tokenizer: return "Error: Model/Tokenizer not loaded."
    caller_id, urgent_msg = "Unknown Caller", "Urgent - Call back ASAP!"
    # The prompt for generate_response should be just the user instruction part
    user_prompt_instruction = (
        f"Generate a Caller ID and an urgent message for a fake emergency notification.\n"
        f"Format:\nCaller: [Caller ID]\nMessage: [Urgent Message]\nOutput ONLY these two lines."
    )
    raw_output = generate_response(model, tokenizer, user_prompt_instruction, device)

    if raw_output.startswith("Error:") or raw_output == "[Empty Response]":
        print(f"Emergency simulation generation failed or empty: {raw_output}")
        return f"🚨 SIMULATED URGENT NOTIFICATION 🚨\n📞 Missed Call: {caller_id}\n💬 Message: \"{urgent_msg}\" (Generation Error)\n🕒 Time: {datetime.now().strftime('%I:%M %p')}"
    try:
        caller_match = re.search(r"Caller:\s*(.*)", raw_output, re.IGNORECASE | re.MULTILINE)
        msg_match = re.search(r"Message:\s*(.*)", raw_output, re.IGNORECASE | re.MULTILINE)

        if caller_match: caller_id = caller_match.group(1).strip() or caller_id
        if msg_match: urgent_msg = msg_match.group(1).strip() or urgent_msg

        # Clean up potential model verbosity
        caller_id = caller_id.split('\n')[0]
        urgent_msg = urgent_msg.split('\n')[0]

    except Exception as e: print(f"WARN: Emergency simulation parsing failed: {e}. Raw output: '{raw_output}'")
    current_time = datetime.now().strftime("%I:%M %p")
    return f"🚨 SIMULATED URGENT NOTIFICATION 🚨\n📞 Missed Call: {caller_id}\n💬 Message: \"{urgent_msg}\"\n🕒 Time: {current_time}"


def add_to_history_df(history_df, new_entry_data, current_id):
    """Adds new entry to history_df and returns the updated df and next id."""
    entry_with_id = {**new_entry_data, 'id': current_id}
    # Ensure all columns from HISTORY_DF_COLUMNS are present in the new entry, adding None if missing
    for col in HISTORY_DF_COLUMNS:
        if col not in entry_with_id:
            entry_with_id[col] = None

    new_df_row = pd.DataFrame([entry_with_id], columns=HISTORY_DF_COLUMNS) # Ensure column order

    # If history_df is empty and has no columns, new_df_row defines them.
    # Otherwise, concat making sure dtypes are compatible or coercible.
    if history_df.empty:
        updated_history_df = new_df_row
    else:
        updated_history_df = pd.concat([history_df, new_df_row], ignore_index=True)

    return updated_history_df, current_id + 1

def save_history_to_csv(history_df, history_csv_path):
    if isinstance(history_df, pd.DataFrame) and not history_df.empty:
        try:
            history_df_to_save = history_df.copy()
            # Ensure correct dtypes before saving
            history_df_to_save['id'] = history_df_to_save['id'].astype(int)
            history_df_to_save['timestamp'] = pd.to_datetime(history_df_to_save['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            history_df_to_save['is_favorite'] = history_df_to_save['is_favorite'].astype(bool)
            history_df_to_save['effectiveness_rating'] = pd.to_numeric(history_df_to_save['effectiveness_rating'], errors='coerce') # Already done, but good check
            history_df_to_save.to_csv(history_csv_path, index=False)
            print(f"History saved to '{history_csv_path}'")
            return True
        except Exception as e:
            print(f"Error saving history: {e}"); traceback.print_exc(); return False
    print("History DataFrame is empty or not a DataFrame. Nothing to save.")
    return False

def toggle_favorite_in_df(history_df, item_id_to_toggle):
    if 'id' in history_df.columns and item_id_to_toggle in history_df['id'].values:
        idx = history_df.index[history_df['id'] == item_id_to_toggle].tolist()
        if idx:
            # Ensure 'is_favorite' column exists and handle potential NaNs before casting to bool
            if 'is_favorite' not in history_df.columns:
                history_df['is_favorite'] = False # Initialize column if it doesn't exist

            current_status_val = history_df.loc[idx[0], 'is_favorite']
            current_status = bool(current_status_val) if pd.notna(current_status_val) else False

            history_df.loc[idx[0], 'is_favorite'] = not current_status
            print(f"Item ID {item_id_to_toggle} favorite status toggled to {not current_status}.")
    else:
        print(f"Item ID {item_id_to_toggle} not found in history DataFrame for toggling favorite.")
    return history_df

def record_feedback_in_df(history_df, item_id_to_rate, rating_val):
    if 'id' in history_df.columns and item_id_to_rate in history_df['id'].values:
        if 0 <= rating_val <= 10:
            idx = history_df.index[history_df['id'] == item_id_to_rate].tolist()
            if idx:
                history_df.loc[idx[0], 'effectiveness_rating'] = rating_val
                print(f"Rating {rating_val}/10 recorded for item ID {item_id_to_rate}.")
        else:
            print(f"Invalid rating value: {rating_val}. Must be between 0 and 10.")
    else:
        print(f"Item ID {item_id_to_rate} not found in history DataFrame for recording feedback.")
    return history_df


def get_ranked_suggestion_list(history_df, situation, priority, plausibility, msg_type, top_n=3):
    suggestions_list = []
    if not isinstance(history_df, pd.DataFrame) or history_df.empty:
        return suggestions_list
    if 'effectiveness_rating' not in history_df.columns or history_df['effectiveness_rating'].isna().all():
        return suggestions_list # No ratings to rank by

    hist_copy = history_df.copy()
    hist_copy['effectiveness_rating'] = pd.to_numeric(hist_copy['effectiveness_rating'], errors='coerce')
    hist_copy['timestamp'] = pd.to_datetime(hist_copy['timestamp'], errors='coerce')

    # Ensure case-insensitive matching for string columns and handle potential None/NaN
    filt = hist_copy[
        (hist_copy['situation'].astype(str).str.lower() == str(situation).lower() if pd.notna(situation) else hist_copy['situation'].isna()) &
        (hist_copy['priority'].astype(str).str.lower() == str(priority).lower() if pd.notna(priority) else hist_copy['priority'].isna()) &
        (hist_copy['plausibility'].astype(str).str.lower() == str(plausibility).lower() if pd.notna(plausibility) else hist_copy['plausibility'].isna()) &
        (hist_copy['message_type'].astype(str).str.lower() == str(msg_type).lower() if pd.notna(msg_type) else hist_copy['message_type'].isna()) &
        (hist_copy['effectiveness_rating'].notna())
    ].copy()

    if filt.empty: return suggestions_list
    # Sort by rating (desc), then by timestamp (desc) to get most recent highly-rated
    ranked = filt.sort_values(by=['effectiveness_rating', 'timestamp'], ascending=[False, False])
    # Get unique texts to avoid showing the exact same suggestion multiple times if it was rated identically multiple times
    unique_texts_df = ranked.drop_duplicates(subset=['generated_text'], keep='first')

    top_suggestions = unique_texts_df.head(top_n)

    for _idx, row in top_suggestions.iterrows():
        suggestions_list.append({"text": row['generated_text'], "rating": f"{row['effectiveness_rating']:.0f}/10"})
    return suggestions_list