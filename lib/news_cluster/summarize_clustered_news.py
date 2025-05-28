# summarize_clustered_news.py
import torch
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datetime import date
import os
import re # For parsing summary

from typing import TypedDict, List

class SummaryRecord(TypedDict):
    title: str
    cluster_id: int
    summarized_news: str  # Full LLM output for the cluster
    date: str

# --- Global Variables for LLM (to avoid reloading) ---
MODEL_ID = "scb10x/typhoon2.1-gemma3-4b"
LLM_PIPELINE = None
LLM_TOKENIZER = None

def initialize_llm_model_and_pipeline():
    """
    Initializes the LLM model and tokenizer if they haven't been already.
    Returns the summarizer pipeline and tokenizer.
    """
    global LLM_PIPELINE, LLM_TOKENIZER
    if LLM_PIPELINE is None:
        print("Initializing LLM model and pipeline (this may take a few minutes)...")
        try:
            LLM_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                device_map="auto",
                # torch_dtype=torch.bfloat16, # or torch.float16
                # load_in_4bit=True, # Optional: for 4-bit quantization
            )
            LLM_PIPELINE = pipeline(
                "text-generation",
                model=model,
                tokenizer=LLM_TOKENIZER,
                # torch_dtype=torch.bfloat16, # or torch.float16
                device_map="auto",
            )
            print("LLM model and pipeline initialized successfully.")
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            LLM_PIPELINE = None # Ensure it's None if initialization fails
            LLM_TOKENIZER = None
    return LLM_PIPELINE, LLM_TOKENIZER

def summarize_news_cluster(cluster_text: str, llm_pipeline, tokenizer_llm) -> str:
    """
    Summarizes a given block of text (representing a news cluster) using the LLM.
    """
    if not llm_pipeline or not tokenizer_llm:
        return "ERROR: LLM Pipeline or Tokenizer not initialized."

    prompt_template = (
        "<|im_start|>system\nคุณเป็นผู้ช่วย AI ที่เชี่ยวชาญด้านการวิเคราะห์ สรุปใจความสำคัญของข่าว และเรียงลำดับเหตุการณ์ตามไทม์ไลน์<|im_end|>\n"
        "<|im_start|>user\n"
        "สำหรับกลุ่มข่าวต่อไปนี้ โปรดดำเนินการดังนี้:\n"
        "1.  **สรุปข่าว:** สรุปใจความสำคัญของกลุ่มข่าวนี้เป็นภาษาไทยอย่างกระชับ\n"
        "2.  **ไทม์ไลน์เหตุการณ์:** เรียงลำดับเหตุการณ์ของข่าวในกลุ่มนี้ให้ชัดเจนที่สุดเท่าที่จะทำได้ หากมีวันที่และเวลาปรากฏในข่าว โปรดระบุด้วยในการเรียงไทม์ไลน์\n\n"
        "--- ข้อมูลข่าวกลุ่มนี้ ---\n"
        f"{cluster_text}\n"
        "--- สิ้นสุดข้อมูลข่าวกลุ่มนี้ ---\n\n"
        "กรุณาตอบตามรูปแบบที่ให้ไว้ด้านล่างนี้ โดยเริ่มจาก **สรุปข่าว:** ตามด้วยเนื้อหาสรุป แล้วต่อด้วย **ไทม์ไลน์เหตุการณ์:** และเนื้อหาไทม์ไลน์:\n" # Modified instruction for clarity
        "**สรุปข่าว:**\n[บทสรุปข่าวของกลุ่มนี้]\n\n"
        "**ไทม์ไลน์เหตุการณ์:**\n[ไทม์ไลน์ของเหตุการณ์ในกลุ่มนี้]\n<|im_end|>\n"
        "<|im_start|>assistant\n" # Keep the assistant start token
    )
    try:
        sequences = llm_pipeline(
            prompt_template,
            max_new_tokens=1500,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            top_k=40,
            eos_token_id=tokenizer_llm.eos_token_id,
            pad_token_id=tokenizer_llm.pad_token_id if tokenizer_llm.pad_token_id is not None else tokenizer_llm.eos_token_id
        )
        
        generated_text = ""
        if sequences and sequences[0]['generated_text']:
            full_output = sequences[0]['generated_text']
            # Extract text after the final <|im_start|>assistant\n
            assistant_marker = "<|im_start|>assistant\n"
            # Find the last occurrence in case the prompt itself had this structure
            last_assistant_marker_idx = full_output.rfind(assistant_marker)

            if last_assistant_marker_idx != -1:
                generated_text = full_output[last_assistant_marker_idx + len(assistant_marker):].strip()
                if generated_text.endswith("<|im_end|>"):
                    generated_text = generated_text[:-len("<|im_end|>")].strip()
            else:
                # Fallback if the exact marker isn't found, try to get text after prompt
                prompt_end_marker = "กรุณาตอบตามรูปแบบที่ให้ไว้ด้านล่างนี้ โดยเริ่มจาก **สรุปข่าว:** ตามด้วยเนื้อหาสรุป แล้วต่อด้วย **ไทม์ไลน์เหตุการณ์:** และเนื้อหาไทม์ไลน์:\n"
                prompt_idx = full_output.find(prompt_end_marker)
                if prompt_idx != -1:
                     generated_text = full_output[prompt_idx + len(prompt_end_marker):].strip() # This might need adjustment
                else:
                    generated_text = "ERROR: Could not parse assistant response. Full output: " + full_output
            return generated_text
        else:
            return "ERROR: LLM did not return any sequences."
            
    except Exception as e:
        return f"EXCEPTION during LLM summarization: {str(e)}"

def parse_summary_from_llm_output(llm_output: str) -> str:
    """
    Extracts the 'สรุปข่าว' part from the LLM's structured output.
    """
    summary_match = re.search(r"\*\*สรุปข่าว:\*\*\s*\n(.*?)(?=\n\n\*\*ไทม์ไลน์เหตุการณ์:\*\*|\Z)", llm_output, re.DOTALL | re.IGNORECASE)
    if summary_match:
        return summary_match.group(1).strip()
    return "ไม่พบส่วนสรุปข่าว"

def generate_dynamic_title(summary_text: str, max_length: int = 70) -> str:
    """
    Generates a concise title from the summary text.
    """
    if not summary_text or summary_text.startswith("ERROR:") or summary_text == "ไม่พบส่วนสรุปข่าว":
        return "หัวข้อสรุปอัตโนมัติ" # Default title if summary is problematic
    
    # Remove "สรุปข่าว:" if it's somehow still there at the beginning
    if summary_text.lower().startswith("**สรุปข่าว:**"):
        summary_text = summary_text[len("**สรุปข่าว:**"):].strip()
    
    first_sentence = summary_text.split('.')[0] # Take up to the first period
    if len(first_sentence) > max_length:
        # Try to find a natural break point (space) before max_length
        break_point = first_sentence.rfind(' ', 0, max_length)
        if break_point != -1:
            return first_sentence[:break_point] + "..."
        else:
            return first_sentence[:max_length] + "..."
    elif first_sentence:
        return first_sentence
    else: # Fallback if first sentence is empty
        return summary_text[:max_length] + "..." if len(summary_text) > max_length else summary_text

def formated_summary(dynamic_title: str, cluster_id: int, full_summarized_content: str, run_date_str: str) -> SummaryRecord:
    return {
        'title': dynamic_title,
        'cluster_id': cluster_id,
        'summarized_news': full_summarized_content, # This is the full LLM output for the cluster
        'date': run_date_str
    }

# --- MODIFIED Function to Export/Append Summary to CSV ---
def export_summary_to_csv(dynamic_title: str, cluster_id: int, full_summarized_content: str, run_date_str: str, csv_filename: str):
    """
    Appends a new summary record (title, cluster_id, summarized_news, date) to a CSV file.
    Creates the file with headers if it doesn't exist.
    """
    data_to_append = {
        'title': [dynamic_title],
        'cluster_id': [cluster_id],
        'summarized_news': [full_summarized_content], # This is the full LLM output for the cluster
        'date': [run_date_str]
    }
    df_new_record = pd.DataFrame(data_to_append)
    
    csv_columns = ['title', 'cluster_id', 'summarized_news', 'date'] # New column order
    file_exists = os.path.exists(csv_filename)

    try:
        if not file_exists:
            df_new_record.to_csv(csv_filename, index=False, encoding='utf-8-sig', columns=csv_columns)
            print(f"Created new CSV file: {csv_filename} and added the first record.")
        else:
            df_new_record.to_csv(csv_filename, mode='a', header=False, index=False, encoding='utf-8-sig', columns=csv_columns)
            # print(f"Appended record to {csv_filename}")
    except Exception as e:
        print(f"Error exporting to CSV '{csv_filename}': {e}")

def process_and_summarize_clustered_news(input_csv_path, output_csv_path: str) -> List[SummaryRecord]:
    """
    Main function to load clustered news, summarize each cluster, and export.
    """
    # LLM_PIPELINE, LLM_TOKENIZER = initialize_llm_model_and_pipeline()
    if not LLM_PIPELINE:
        print("LLM initialization failed. Exiting.")
        return

    current_processing_date = date.today().strftime("%Y-%m-%d")

    if type(input_csv_path) == str:
        print(f"\n📥 Reading clustered news from: {input_csv_path}")
        try:
            clustered_df = pd.read_csv(input_csv_path)
        except FileNotFoundError:
            print(f"❌ Error: Input CSV file not found at {input_csv_path}")
            return
        except Exception as e:
            print(f"❌ Error reading CSV {input_csv_path}: {e}")
            return
    else:
        clustered_df = input_csv_path.copy()

    if 'content' not in clustered_df.columns or 'cluster' not in clustered_df.columns:
        print("❌ Error: Input CSV must contain 'content' and 'cluster' columns.")
        return

    unique_cluster_ids = sorted(clustered_df['cluster'].unique())

    print(f"\n--- Starting News Summarization and Export to {output_csv_path} ---")
    print(f"processing {len(unique_cluster_ids)} clusters...")

    to_return = []
    
    for cluster_id_val in unique_cluster_ids:
        cluster_id_val = int(cluster_id_val)  # Ensure cluster ID is an integer
        if cluster_id_val == -1:
            print(f"\nSkipping noise cluster (Cluster ID: {cluster_id_val}) for main summarization.")
            # export_summary_to_csv("กลุ่มข่าว Noise",
            #                         cluster_id_val,
            #                         "ข่าวในกลุ่ม Noise (ไม่ได้ทำการสรุปด้วย LLM)",
            #                         current_processing_date,
            #                         output_csv_path)
            to_return.append(formated_summary(
                "กลุ่มข่าว Noise",
                cluster_id_val,
                "ข่าวในกลุ่ม Noise (ไม่ได้ทำการสรุปด้วย LLM)",
                current_processing_date,
            ))
            continue

        print(f"\n--- Processing Cluster ID: {cluster_id_val} ---")
        
        cluster_articles_df = clustered_df[clustered_df['cluster'] == cluster_id_val].dropna(subset=['content'])
        
        if cluster_articles_df.empty:
            print(f"No valid content found for cluster {cluster_id_val}.")
            # export_summary_to_csv(f"กลุ่ม {cluster_id_val}: ไม่มีเนื้อหา",
            #                       cluster_id_val,
            #                       "ไม่พบข่าวในกลุ่มนี้สำหรับสรุป (หรือเนื้อหาข่าวเป็นค่าว่าง)",
            #                       current_processing_date,
            #                       output_csv_path)
            to_return.append(formated_summary(
                f"กลุ่ม {cluster_id_val}: ไม่มีเนื้อหา",
                cluster_id_val,
                "ไม่พบข่าวในกลุ่มนี้สำหรับสรุป (หรือเนื้อหาข่าวเป็นค่าว่าง)",
                current_processing_date,
            ))
            continue
            
        current_cluster_sentences = cluster_articles_df['content'].tolist()
        combined_text_for_llm = "\n\n---\n\n".join(map(str, current_cluster_sentences))
        
        if not combined_text_for_llm.strip():
            print(f"Combined text for cluster {cluster_id_val} is empty after stripping.")
            # export_summary_to_csv(f"กลุ่ม {cluster_id_val}: เนื้อหาว่างเปล่า",
            #                       cluster_id_val,
            #                       f"Cluster {cluster_id_val}: เนื้อหาข่าวว่างเปล่า ไม่สามารถสรุปได้",
            #                       current_processing_date,
            #                       output_csv_path)
            to_return.append(formated_summary(
                f"กลุ่ม {cluster_id_val}: เนื้อหาว่างเปล่า",
                cluster_id_val,
                f"Cluster {cluster_id_val}: เนื้อหาข่าวว่างเปล่า ไม่สามารถสรุปได้",
                current_processing_date,
            ))
            continue
            
        print(f"Summarizing text for cluster {cluster_id_val}...")
        # llm_output_full is the entire response from the LLM, should include summary and timeline
        llm_output_full = summarize_news_cluster(combined_text_for_llm, LLM_PIPELINE, LLM_TOKENIZER)
        
        # Extract the summary part for the dynamic title
        parsed_summary_only = parse_summary_from_llm_output(llm_output_full)
        dynamic_csv_title = generate_dynamic_title(parsed_summary_only)
        
        print(f"Generated Title for CSV: {dynamic_csv_title}")
        print(f"Full LLM Output for Cluster {cluster_id_val}:\n{llm_output_full}")
        
        # export_summary_to_csv(dynamic_csv_title,
        #                       cluster_id_val,
        #                       llm_output_full, # Save the full LLM output (summary + timeline)
        #                       current_processing_date,
        #                       output_csv_path)
        to_return.append(formated_summary(
            dynamic_csv_title,
            cluster_id_val,
            llm_output_full, # Save the full LLM output (summary + timeline)
            current_processing_date,
        ))
    
    print(f"\n--- All processing finished. Summaries appended/saved to: {output_csv_path} ---")

    return to_return  # Return the list of summaries for further processing if needed
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize clustered news using LLM and export to CSV with dynamic titles.")
    parser.add_argument("input_clustered_csv", 
                        help="Path to the input CSV file from news_cluster.py (e.g., news_clustered.csv)")
    parser.add_argument("--output_summary_csv", 
                        default="llm_summaries_archive_with_dynamic_title.csv", 
                        help="Path to the output CSV file for summaries (default: llm_summaries_archive_with_dynamic_title.csv)")

    args = parser.parse_args()
    process_and_summarize_clustered_news(args.input_clustered_csv, args.output_summary_csv)