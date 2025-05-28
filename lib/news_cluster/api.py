# api.py
import uvicorn
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
import asyncio # Though not directly used for execution, good for async context of FastAPI
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse
import pandas as pd
import io
from embed_news import Embeder

# --- Configuration ---
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8002")) # Using a different port for this specific API
DEBUG = os.getenv("DEBUG", "true").lower() == "true"

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="News Summarization API (v2 - Dynamic Titles)",
    description="API to trigger the summarization of clustered news articles and export to a CSV with dynamic titles and cluster IDs.",
    version="2.1.0" # Updated version
)

# --- CORS Configuration (Allow all for NGROK/Frontend development convenience) ---
origins = ["*"] # Allows all origins - BE CAREFUL IN PRODUCTION
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    from summarize_clustered_news import (
        initialize_llm_model_and_pipeline,
        process_and_summarize_clustered_news,
    )
    summarizer_script_available = True
    
except ImportError as e:
    logging.error(f"Failed to import from summarize_clustered_news.py: {e}. API endpoints will not function correctly.")
    summarizer_script_available = False
    # Define dummy functions if import fails, so API can start but endpoints will error out
    def initialize_llm_model_and_pipeline(): logging.error("Summarizer script not found, dummy initialize_llm called"); return None, None
    def process_and_summarize_clustered_news(input_path, output_path): logging.error("Summarizer script not found, dummy process_and_summarize called"); return None
    SUMMARIZER_LLM_PIPELINE = None
    SUMMARIZER_LLM_TOKENIZER = None

# --- Pydantic Models ---
# class SummarizationRequest(BaseModel):
#     input_clustered_csv: str = Field(..., example="news_clustered.csv", description="Path to the CSV file from news_cluster.py (must have 'content' and 'cluster' columns).")
#     # Aligning default with the summarizer script's default
#     output_summary_csv: Optional[str] = Field("llm_summaries_archive_with_dynamic_title.csv", 
#                                              example="llm_summaries_with_titles.csv", 
#                                              description="Desired path for the output CSV file containing summaries.")

# class SummarizationResponse(BaseModel):
#     message: str
#     output_file: Optional[str] = None
#     status: str = "pending" # pending, processing, completed, failed

# class EmbedRequest(BaseModel):
#     input_clustered_csv: str = Field(..., example="news_clustered.csv", description="Path to the CSV file from news_cluster.py (must have 'content' and 'cluster' columns).")
#     # Aligning default with the summarizer script's default
#     output_summary_csv: Optional[str] = Field("llm_summaries_archive_with_dynamic_title.csv", 
#                                              example="llm_summaries_with_titles.csv", 
                           
#                                              description="Desired path for the output CSV file containing summaries.")
# class EmbedResponse(BaseModel):
#     message: str
#     output_file: Optional[str] = None
#     status: str = "pending" # pending, processing, completed, failed

# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    global embedder
    embedder = Embeder()
    global SUMMARIZER_LLM_PIPELINE, SUMMARIZER_LLM_TOKENIZER
    SUMMARIZER_LLM_PIPELINE, SUMMARIZER_LLM_TOKENIZER = initialize_llm_model_and_pipeline()
    # Attempt to import functions from your summarization script
    logger.info(f"News Summarization API (v{app.version}) starting up...")
    if summarizer_script_available:
        initialize_llm_model_and_pipeline() # This function now handles the global LLM_PIPELINE
        if SUMMARIZER_LLM_PIPELINE and SUMMARIZER_LLM_TOKENIZER:
            logger.info("Summarizer LLM model initialization check complete on startup.")
        else:
            logger.error("Summarizer LLM model initialization FAILED on startup. /summarize-news/ will not function correctly.")
    else:
        logger.error("Summarizer script (summarize_clustered_news.py) not found. LLM will not be initialized.")

# def run_summarization_background(input_path: str, output_path: str):
#     """
#     This function will be run in the background.
#     It calls the potentially long-running summarization process from summarize_clustered_news.py.
#     """
#     logger.info(f"Background task started: Summarizing {input_path} to {output_path}")
#     try:
#         # Call the main processing function from the imported script
#         process_and_summarize_clustered_news(input_path, output_path)
#         logger.info(f"Background task finished: Summarization for {input_path} completed. Output should be at {output_path}")
#     except Exception as e:
#         logger.error(f"Error in background summarization task for {input_path}: {e}", exc_info=True)
#         # For more robust error reporting, you might write status to a file or DB
#         # that the main API could poll or another endpoint could check.

@app.post("/embed/")
async def embed(file: UploadFile = File(...), repo_name: str = Form(...)):
    df = pd.read_csv(file.file)

    # Example: Apply operation based on string param
    pkl_path = embedder.embed_colbert(df,repo_name,api=True)
    # output = io.StringIO()
    # df.to_csv(output, index=False)
    # output.seek(0)
    # return StreamingResponse(output, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=processed.csv"})
    return FileResponse(
        path=pkl_path,
        media_type="application/octet-stream",
        filename=pkl_path
    )

@app.post("/summarize-news/")
async def summarize(file: UploadFile = File(...), output_path: str = Form(...)):
    # ✅ Step 1: Read uploaded pickle file into a DataFrame
    # df = pd.read_pickle(file.file)
    df = pd.read_csv(file.file)

    # ✅ Step 2: Do processing (optional, plug in your own logic)
    # Example: Maybe reverse your embedding here?
    process_and_summarize_clustered_news(df, output_path)

    # ✅ Step 3: Return CSV file
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={output_path}"}
    )

# @app.post("/summarize-news/", response_model=SummarizationResponse, tags=["Summarization"])
# async def trigger_summarization(request: SummarizationRequest, background_tasks: BackgroundTasks):
#     """
#     Triggers the news summarization pipeline. The output CSV will contain:
#     'title' (dynamically generated short summary), 'cluster_id', 
#     'summarized_news' (full LLM output including detailed summary and timeline), and 'date'.
#     The summarization runs as a background task.
#     """
#     logger.info(f"Received request to summarize: {request.input_clustered_csv} -> {request.output_summary_csv}")

#     if not summarizer_script_available:
#         logger.error("Summarization trigger failed: Core summarizer script not found.")
#         raise HTTPException(status_code=503, detail="Summarization service is unavailable: core script not found.")

#     if SUMMARIZER_LLM_PIPELINE is None or SUMMARIZER_LLM_TOKENIZER is None:
#         logger.warning("LLM for summarizer not ready, attempting to re-initialize (should have happened at startup)...")
#         initialize_llm_model_and_pipeline() 
#         if SUMMARIZER_LLM_PIPELINE is None or SUMMARIZER_LLM_TOKENIZER is None:
#             logger.error("Summarization trigger failed: LLM service for summarization is not ready or failed to initialize.")
#             raise HTTPException(status_code=503, detail="LLM service for summarization is not ready or failed to initialize.")

#     input_file = request.input_clustered_csv
#     output_file = request.output_summary_csv # This is the filename that will be used by the background task.

#     if not os.path.exists(input_file):
#         logger.error(f"Input file not found: {input_file}")
#         raise HTTPException(status_code=404, detail=f"Input clustered CSV file not found: {input_file}")

#     background_tasks.add_task(run_summarization_background, input_file, output_file)
    
#     logger.info(f"Summarization task for '{input_file}' added to background. Output will be at '{output_file}'.")
#     return SummarizationResponse(
#         message="Summarization process has been started in the background. The output CSV will be generated with dynamic titles and cluster IDs. Check server logs and the specified output file for completion.",
#         output_file=output_file,
#         status="processing"
#     )

@app.get("/health", tags=["Health"])
async def health_check():
    if not summarizer_script_available:
        return {"status": "error", "message": "Summarization API is partially unavailable: Core summarizer script not found."}
    if SUMMARIZER_LLM_PIPELINE and SUMMARIZER_LLM_TOKENIZER:
        return {"status": "ok", "message": f"Summarization API (v{app.version}) is running and LLM model is loaded."}
    else:
        return {"status": "warning", "message": f"Summarization API (v{app.version}) is running, but LLM model/pipeline is not initialized. Check startup logs."}

if __name__ == "__main__":
    logger.info(f"Starting Uvicorn server on {API_HOST}:{API_PORT} for News Summarization API (v{app.version})...")
    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=DEBUG)