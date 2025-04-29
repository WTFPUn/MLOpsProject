import pandas as pd
from llama_cpp import Llama
import time

data = pd.read_csv("../scraping/thairath_news.csv")[["content"]].dropna()
test = data["content"].values.tolist()
model_path = "model/pathumma-llm-text-1.0.0-q4_k_m.gguf"  # <--- CHANGE THIS
# 2. Initialize the Llama model
#    Adjust n_gpu_layers based on your hardware and desired GPU offloading:
#    - 0: CPU only
#    - -1: Offload all possible layers to GPU (recommended if you have a powerful GPU)
#    - Positive number: Offload that many layers to GPU
#    Adjust n_ctx based on the model's context window size (check model card)
try:
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,        # Set to 0 for CPU-only, -1 for max GPU offload
        n_ctx=2048,             # Model's context size (adjust as needed)
        verbose=True            # Set to False to reduce log output
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def label(prompt):
    # 4. Run inference
    print("\n--- Generating Response ---")
    start_time = time.time() 
    try:
        output = llm(
            prompt,
            max_tokens=250,       # Max number of tokens to generate
            stop=["\n", "User:", "Assistant:"], # Optional: Stop generation sequences
            echo=False            # Set to True to include the prompt in the output
        )
        generated_text = output["choices"][0]["text"].strip()

        end_time = time.time() # Optional: Stop timer
        duration = end_time - start_time # Optional: Calculate duration

        # 5. Print the generated text
        print(generated_text)
        print(f"\n--- Generation finished in {duration:.2f} seconds ---") # Optional
        return generated_text

    except Exception as e:
        print(f"Error during inference: {e}")
        return None

labels = []
for i in test:
    # 3. Define your prompt
    prompt = f"จากข่าวนี้จงสรุปหัวข้อที่เกี่ยวข้องออกมาเป็นชื่อบุคคลหลัก และ เหตุการณ์ในเนื้อข่าว ข่าว: {i}"
    t = label(prompt)
    labels.append(t)

# save result
data["llm_label"] = labels
data.to_csv(f"llm_labeled_{len(labels)}c.csv")
