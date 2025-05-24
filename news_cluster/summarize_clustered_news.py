import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datetime import date
import os

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
                torch_dtype=torch.bfloat16, # or torch.float16
                # load_in_4bit=True, # Optional: for 4-bit quantization
            )
            LLM_PIPELINE = pipeline(
                "text-generation",
                model=model,
                tokenizer=LLM_TOKENIZER,
                torch_dtype=torch.bfloat16, # or torch.float16
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
        "กรุณาตอบตามรูปแบบ:\n"
        "**สรุปข่าว:**\n[บทสรุปข่าวของกลุ่มนี้]\n\n"
        "**ไทม์ไลน์เหตุการณ์:**\n[ไทม์ไลน์ของเหตุการณ์ในกลุ่มนี้]\n<|im_end|>\n"
        "<|im_start|>assistant\n"
        "**สรุปข่าว:**\n"
    )
    try:
        sequences = llm_pipeline(
            prompt_template,
            max_new_tokens=1500, # Increased for potentially detailed output
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            top_k=50,
            eos_token_id=tokenizer_llm.eos_token_id,
            pad_token_id=tokenizer_llm.pad_token_id if tokenizer_llm.pad_token_id is not None else tokenizer_llm.eos_token_id
        )
        
        generated_summary = ""
        if sequences and sequences[0]['generated_text']:
            full_output = sequences[0]['generated_text']
            assistant_marker = "<|im_start|>assistant\n"
            start_idx = full_output.find(assistant_marker)
            
            if start_idx != -1:
                generated_summary = full_output[start_idx + len(assistant_marker):].strip()
                if generated_summary.endswith("<|im_end|>"):
                    generated_summary = generated_summary[:-len("<|im_end|>")].strip()
            else:
                generated_summary = "ERROR: Could not parse assistant response. Full output: " + full_output
            return generated_summary
        else:
            return "ERROR: LLM did not return any sequences."
            
    except Exception as e:
        return f"EXCEPTION during LLM summarization: {str(e)}"

def export_summary_to_csv(title_content: str, summarized_news_content: str, run_date_str: str, csv_filename: str):
    """
    Appends a new summary record (title, summarized_news, date) to a CSV file.
    Creates the file with headers if it doesn't exist.
    """
    data_to_append = {
        'title': [title_content],
        'summarized_news': [summarized_news_content],
        'date': [run_date_str]
    }
    df_new_record = pd.DataFrame(data_to_append)
    
    csv_columns = ['title', 'summarized_news', 'date']
    file_exists = os.path.exists(csv_filename)

    try:
        if not file_exists:
            df_new_record.to_csv(csv_filename, index=False, encoding='utf-8-sig', columns=csv_columns)
            print(f"Created new CSV file: {csv_filename} and added the first record.")
        else:
            df_new_record.to_csv(csv_filename, mode='a', header=False, index=False, encoding='utf-8-sig', columns=csv_columns)
            # print(f"Appended record to {csv_filename}") # Uncomment for verbose output
    except Exception as e:
        print(f"Error exporting to CSV '{csv_filename}': {e}")

def process_and_summarize_clustered_news(input_csv_path: str, output_csv_path: str):
    """
    Main function to load clustered news, summarize each cluster, and export.
    """
    # Initialize LLM (do this once)
    summarizer_pipeline, tokenizer_for_llm = initialize_llm_model_and_pipeline()
    if not summarizer_pipeline:
        print("LLM initialization failed. Exiting.")
        return

    current_processing_date = date.today().strftime("%Y-%m-%d")

    print(f"\n📥 Reading clustered news from: {input_csv_path}")
    try:
        clustered_df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"❌ Error: Input CSV file not found at {input_csv_path}")
        return
    except Exception as e:
        print(f"❌ Error reading CSV {input_csv_path}: {e}")
        return

    # Ensure 'content' and 'cluster' columns exist
    if 'content' not in clustered_df.columns or 'cluster' not in clustered_df.columns:
        print("❌ Error: Input CSV must contain 'content' and 'cluster' columns.")
        return

    unique_cluster_ids = sorted(clustered_df['cluster'].unique())

    print(f"\n--- Starting News Summarization and Export to {output_csv_path} ---")

    for cluster_id in unique_cluster_ids:
        summary_title = f"สรุปข่าวกลุ่มที่ {cluster_id}"

        if cluster_id == -1:
            print(f"\nSkipping noise cluster (Cluster ID: {cluster_id}) for main summarization.")
            # Optional: Log noise, or write a placeholder to the CSV
            # export_summary_to_csv(summary_title,
            #                         "ข่าวในกลุ่ม Noise (ไม่ได้ทำการสรุปด้วย LLM)",
            #                         current_processing_date,
            #                         output_csv_path)
            continue

        print(f"\n--- Processing Cluster ID: {cluster_id} ---")
        
        # Filter DataFrame for the current cluster
        # Ensure 'content' is not NaN before trying to join
        cluster_articles_df = clustered_df[clustered_df['cluster'] == cluster_id].dropna(subset=['content'])
        
        if cluster_articles_df.empty:
            print(f"No valid content found for cluster {cluster_id}.")
            export_summary_to_csv(summary_title,
                                  "ไม่พบข่าวในกลุ่มนี้สำหรับสรุป (หรือเนื้อหาข่าวเป็นค่าว่าง)",
                                  current_processing_date,
                                  output_csv_path)
            continue
            
        current_cluster_sentences = cluster_articles_df['content'].tolist()
        combined_text_for_llm = "\n\n---\n\n".join(map(str, current_cluster_sentences)) # Ensure all parts are strings
        
        if not combined_text_for_llm.strip():
            print(f"Combined text for cluster {cluster_id} is empty after stripping.")
            export_summary_to_csv(summary_title,
                                  f"Cluster {cluster_id}: เนื้อหาข่าวว่างเปล่า ไม่สามารถสรุปได้",
                                  current_processing_date,
                                  output_csv_path)
            continue
            
        print(f"Summarizing text for cluster {cluster_id}...")
        llm_output_summary = summarize_news_cluster(combined_text_for_llm, summarizer_pipeline, tokenizer_for_llm)
        
        print(f"LLM Output for Cluster {cluster_id}:\n{llm_output_summary}")
        
        export_summary_to_csv(summary_title,
                              llm_output_summary, # This contains the full structured output
                              current_processing_date,
                              output_csv_path)
    
    print(f"\n--- All processing finished. Summaries appended/saved to: {output_csv_path} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize clustered news using LLM and export to CSV.")
    parser.add_argument("input_clustered_csv", 
                        help="Path to the input CSV file from news_cluster.py (e.g., news_clustered.csv)")
    parser.add_argument("--output_summary_csv", 
                        default="llm_summaries_archive_with_title.csv", 
                        help="Path to the output CSV file for summaries (default: llm_summaries_archive_with_title.csv)")

    args = parser.parse_args()
    process_and_summarize_clustered_news(args.input_clustered_csv, args.output_summary_csv)