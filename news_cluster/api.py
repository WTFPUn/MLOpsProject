import uvicorn
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
import asyncio
from typing import Optional

# Attempt to import functions from your summarization script
try:
    from summarize_clustered_news import (
        initialize_llm_model_and_pipeline,
        process_and_summarize_clustered_news,
        LLM_PIPELINE as SUMMARIZER_LLM_PIPELINE, # Use an alias to avoid name clashes if any
        LLM_TOKENIZER as SUMMARIZER_LLM_TOKENIZER
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


# --- Configuration ---
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8002")) # Using a different port
DEBUG = os.getenv("DEBUG", "true").lower() == "true"

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="News Summarization Trigger API",
    description="API to trigger the summarization of clustered news articles and export to CSV.",
    version="1.0.0"
)

# --- CORS Configuration (Allow all for NGROK/Frontend development convenience) ---
origins = ["*"] # Allows all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class SummarizationRequest(BaseModel):
    input_clustered_csv: str = Field(..., example="news_clustered.csv", description="Path to the CSV file from news_cluster.py (must have 'content' and 'cluster' columns).")
    output_summary_csv: Optional[str] = Field("llm_summaries_output.csv", example="llm_summaries.csv", description="Desired path for the output CSV file containing summaries.")

class SummarizationResponse(BaseModel):
    message: str
    output_file: Optional[str] = None
    error: Optional[str] = None
    status: str = "pending" # pending, processing, completed, failed


# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    logger.info("News Summarization API starting up...")
    if summarizer_script_available:
        # Initialize LLM model and pipeline on startup.
        # The initialize_llm_model_and_pipeline function should handle not re-initializing if already done.
        initialize_llm_model_and_pipeline()
        if SUMMARIZER_LLM_PIPELINE and SUMMARIZER_LLM_TOKENIZER:
            logger.info("Summarizer LLM model initialization check complete on startup.")
        else:
            logger.error("Summarizer LLM model initialization FAILED on startup.")
    else:
        logger.error("Summarizer script (summarize_clustered_news.py) not found. LLM will not be initialized.")

def run_summarization_background(input_path: str, output_path: str):
    """
    This function will be run in the background.
    It calls the potentially long-running summarization process.
    """
    logger.info(f"Background task started: Summarizing {input_path} to {output_path}")
    try:
        # This is a blocking call, but it's running in a background thread managed by FastAPI
        process_and_summarize_clustered_news(input_path, output_path)
        logger.info(f"Background task finished: Summarization for {input_path} completed. Output: {output_path}")
    except Exception as e:
        logger.error(f"Error in background summarization task for {input_path}: {e}", exc_info=True)
        # Note: Error handling here won't directly reflect in the API response
        # for background tasks. You'd need a more sophisticated status tracking mechanism.


@app.post("/summarize-news/", response_model=SummarizationResponse, tags=["Summarization"])
async def trigger_summarization(request: SummarizationRequest, background_tasks: BackgroundTasks):
    """
    Triggers the news summarization pipeline.
    The summarization will run as a background task.
    """
    logger.info(f"Received request to summarize: {request.input_clustered_csv} -> {request.output_summary_csv}")

    if not summarizer_script_available:
        raise HTTPException(status_code=503, detail="Summarization service is unavailable: core script not found.")

    if SUMMARIZER_LLM_PIPELINE is None or SUMMARIZER_LLM_TOKENIZER is None:
        logger.warning("LLM for summarizer not ready, attempting to initialize...")
        initialize_llm_model_and_pipeline() # Attempt to initialize if not done during startup
        if SUMMARIZER_LLM_PIPELINE is None or SUMMARIZER_LLM_TOKENIZER is None:
            raise HTTPException(status_code=503, detail="LLM service for summarization is not ready or failed to initialize.")

    input_file = request.input_clustered_csv
    output_file = request.output_summary_csv

    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        raise HTTPException(status_code=404, detail=f"Input clustered CSV file not found: {input_file}")

    # Add the summarization process to background tasks
    background_tasks.add_task(run_summarization_background, input_file, output_file)
    
    logger.info(f"Summarization task for '{input_file}' added to background. Output will be '{output_file}'.")
    return SummarizationResponse(
        message="Summarization process has been started in the background. Check server logs and output file for completion.",
        output_file=output_file, # Returns the expected output path
        status="processing"
    )

@app.get("/health", tags=["Health"])
async def health_check():
    if not summarizer_script_available:
        return {"status": "error", "message": "Summarization API is partially unavailable: Core summarizer script not found."}
    if SUMMARIZER_LLM_PIPELINE and SUMMARIZER_LLM_TOKENIZER:
        return {"status": "ok", "message": "Summarization API is running and LLM model is loaded."}
    else:
        return {"status": "warning", "message": "Summarization API is running, but LLM model/pipeline is not initialized. Try calling /summarize-news/ to trigger initialization or check startup logs."}

if __name__ == "__main__":
    # The LLM initialization is handled by the FastAPI startup event.
    logger.info(f"Starting Uvicorn server on {API_HOST}:{API_PORT} for News Summarization API...")
    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=DEBUG)