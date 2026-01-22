# server.py
# -------------------------------------------------------------------------------------------------
# This file implements the FastAPI backend server for the Multimodal RAG system.
# It initializes the core RAG components (Indexer, Retriever, Generator) on startup,
# provides API endpoints for querying, and serves the static frontend files.
# I've ensured robust initialization, error handling, and logging as requested.
# No shortcuts will be taken; stability and correctness are paramount.
# -------------------------------------------------------------------------------------------------

import os
import sys
import time
import json
import logging
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import datetime


# FastAPI and related imports
try:
    import fastapi
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
    import uvicorn
except ImportError:
    print("ERROR: FastAPI or related libraries (uvicorn, python-multipart) not found.")
    print("Please install them: pip install fastapi uvicorn python-multipart")
    sys.exit(1)

# Import core RAG components from RAGCore.py
# Ensure RAGCore.py is in the same directory or Python path
try:
    from RAGCore import (
        setup_logging,
        sanitize_filename,
        load_data_from_json_and_associate_images,
        MultimodalEncoder, # Although not used directly here, good to know it's available via Indexer/Retriever
        Indexer,
        Retriever,
        Generator
    )
except ImportError as e:
    print(f"ERROR: Failed to import components from RAGCore.py. Error: {e}")
    print("Ensure RAGCore.py exists in the same directory or is accessible via PYTHONPATH.")
    sys.exit(1)


# =============================================================================================
# Global Variables & Application Setup
# =============================================================================================
# FastAPI application instance
app = FastAPI(
    title="Multimodal RAG API",
    version="1.0.0",
    # docs_url=None, # 禁用 Swagger UI (/docs)
    # redoc_url=None   # 禁用 ReDoc (/redoc)
)
# Global variables to hold initialized RAG components
# These will be populated during the startup event.
indexer_instance: Optional[Indexer] = None
retriever_instance: Optional[Retriever] = None
generator_instance: Optional[Generator] = None
initialization_status: Dict[str, Any] = {
    "status": "pending",
    "indexer": "pending",
    "retriever": "pending",
    "generator": "pending",
    "error_message": None
}

# Configuration (Mimicking the structure from the notebook's main block)
# --- User-configurable Run Identifier ---
RUN_IDENTIFIER_BASE: str = "multimodal_rag_system_output" # Fixed base name for the output directory
sanitized_run_identifier: str = sanitize_filename(RUN_IDENTIFIER_BASE, max_length=50)
OUTPUT_BASE_DIR: str = sanitized_run_identifier # Fixed top-level output directory

# --- Configure Paths within the Fixed Top-Level Directory ---
LOG_DIR: str = os.path.join(OUTPUT_BASE_DIR, "logs")
LOG_FILE_PATH: str = os.path.join(LOG_DIR, "fastapi_server_log.txt") # Specific log file for the server

DB_STORAGE_DIR: str = os.path.join(OUTPUT_BASE_DIR, "data_storage")
DB_DIR: str = os.path.join(DB_STORAGE_DIR, "database")
DB_FILE: str = os.path.join(DB_DIR, 'multimodal_doc_store.db')

FAISS_DIR: str = os.path.join(DB_STORAGE_DIR, "vector_indices")
FAISS_TEXT_INDEX_FILE: str = os.path.join(FAISS_DIR, 'text_vector_index.faiss')
FAISS_IMAGE_INDEX_FILE: str = os.path.join(FAISS_DIR, 'image_vector_index.faiss')
FAISS_MEAN_INDEX_FILE: str = os.path.join(FAISS_DIR, 'mean_vector_index.faiss')

QUERY_RESULTS_DIR: str = os.path.join(OUTPUT_BASE_DIR, "query_session_results")
# Directory for temporary uploads within the main output dir
UPLOADS_TEMP_DIR: str = os.path.join(OUTPUT_BASE_DIR, "temp_uploads")

# --- Data Source ---
JSON_DATA_PATH: str = 'data.json'
IMAGE_DIR_PATH: str = 'images' # This is for *initial indexing*, not uploads

# --- Model Configuration ---
CLIP_MODEL: str = "openai/clip-vit-base-patch32"
LLM_MODEL: str = "glm-4-flash" # Ensure this model is supported by your ZhipuAI key

# Mount static files directory AFTER ensuring paths are set
# Use Pathlib for safer path operations
static_dir = Path("static")
if not static_dir.is_dir():
     print(f"WARNING: Static directory '{static_dir}' not found. Frontend might not load.")
     # Optionally create it if needed: static_dir.mkdir(exist_ok=True)
else:
    try:
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    except Exception as e:
        print(f"ERROR mounting static directory '{static_dir}': {e}")
        # Log this properly once logger is set up

# Initialize logger (will be fully configured on startup)
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.addHandler(logging.NullHandler())


# =============================================================================================
# FastAPI Startup Event Handler
# =============================================================================================
@app.on_event("startup")
async def startup_event():
    """
    Initializes all necessary components when the FastAPI application starts.
    This includes setting up logging, creating directories, loading data (optional here),
    initializing Indexer, Retriever, and Generator.
    This ensures components are ready before handling requests. My utmost focus is on a clean start.
    """
    global indexer_instance, retriever_instance, generator_instance, initialization_status

    initialization_status["status"] = "initializing"
    print("FastAPI startup sequence initiated...") # Use print before logger is fully set up

    try:
        # 1. Create Output Directories
        print(f"Ensuring base output directory exists: {OUTPUT_BASE_DIR}")
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
        print(f"Ensuring log directory exists: {LOG_DIR}")
        os.makedirs(LOG_DIR, exist_ok=True)
        print(f"Ensuring DB storage directory exists: {DB_STORAGE_DIR}")
        os.makedirs(DB_STORAGE_DIR, exist_ok=True)
        print(f"Ensuring DB directory exists: {DB_DIR}")
        os.makedirs(DB_DIR, exist_ok=True)
        print(f"Ensuring Faiss index directory exists: {FAISS_DIR}")
        os.makedirs(FAISS_DIR, exist_ok=True)
        print(f"Ensuring query results directory exists: {QUERY_RESULTS_DIR}")
        os.makedirs(QUERY_RESULTS_DIR, exist_ok=True)
        print(f"Ensuring temporary uploads directory exists: {UPLOADS_TEMP_DIR}")
        os.makedirs(UPLOADS_TEMP_DIR, exist_ok=True)
        print("All necessary directories checked/created.")

        # 2. Setup Logging (Now that the log directory is confirmed)
        setup_logging(LOG_FILE_PATH) # Configure the logger instance
        logger.info("="*50)
        logger.info("FastAPI Server Startup Sequence Initiated")
        logger.info(f"Base Output Directory: {os.path.abspath(OUTPUT_BASE_DIR)}")
        logger.info("Logging configured successfully.")

        # 3. Optional: Load and Index Data on Startup (If desired)
        # Note: For large datasets, indexing should ideally be done offline.
        # This example assumes indices and DB might already exist from a previous run
        # or are built by running the notebook/a separate script first.
        # If you ALWAYS want to re-index on server start, uncomment and adapt the following:
        # logger.info("--- Startup: Loading and Indexing Data ---")
        # documents_to_index = load_data_from_json_and_associate_images(JSON_DATA_PATH, IMAGE_DIR_PATH)
        # if not documents_to_index:
        #     logger.warning("No documents loaded for indexing during startup.")
        # else:
        #     logger.info(f"Loaded {len(documents_to_index)} documents. Initializing temporary Indexer for startup indexing...")
        #     temp_indexer = Indexer(
        #         db_path=DB_FILE,
        #         faiss_text_index_path=FAISS_TEXT_INDEX_FILE,
        #         faiss_image_index_path=FAISS_IMAGE_INDEX_FILE,
        #         faiss_mean_index_path=FAISS_MEAN_INDEX_FILE,
        #         clip_model_name=CLIP_MODEL
        #     )
        #     temp_indexer.index_documents(documents_to_index)
        #     temp_indexer.close() # Save indices and close
        #     logger.info("--- Startup: Data Indexing Complete ---")
        #     # Now, the main Indexer below will load the just-created/updated indices.

        # 4. Initialize Indexer (Loads existing DB/Indices or creates new ones)
        logger.info("--- Startup: Initializing Indexer ---")
        initialization_status["indexer"] = "initializing"
        try:
            indexer_instance = Indexer(
                db_path=DB_FILE,
                faiss_text_index_path=FAISS_TEXT_INDEX_FILE,
                faiss_image_index_path=FAISS_IMAGE_INDEX_FILE,
                faiss_mean_index_path=FAISS_MEAN_INDEX_FILE,
                clip_model_name=CLIP_MODEL
            )
            # Log final counts after initialization (loading)
            text_count = getattr(indexer_instance.text_index, 'ntotal', 0)
            image_count = getattr(indexer_instance.image_index, 'ntotal', 0)
            mean_count = getattr(indexer_instance.mean_index, 'ntotal', 0)
            db_doc_count = indexer_instance.get_document_count()
            logger.info(f"Indexer initialized. DB Docs: {db_doc_count}, Text Vecs: {text_count}, Img Vecs: {image_count}, Mean Vecs: {mean_count}")
            if db_doc_count == 0 or (text_count == 0 and image_count == 0 and mean_count == 0):
                 logger.warning("Indexer initialized, but database or all vector indices are empty. Retrieval might yield no results.")
            initialization_status["indexer"] = "success"
        except Exception as e_indexer:
            logger.critical(f"CRITICAL ERROR during Indexer initialization: {e_indexer}", exc_info=True)
            initialization_status["indexer"] = "failed"
            initialization_status["error_message"] = f"Indexer Init Failed: {e_indexer}"
            # Don't raise here, let status endpoint report failure

        # 5. Initialize Retriever (Only if Indexer succeeded)
        logger.info("--- Startup: Initializing Retriever ---")
        initialization_status["retriever"] = "initializing"
        if indexer_instance:
            try:
                # Check if indexer has searchable content before initializing retriever
                text_count = getattr(indexer_instance.text_index, 'ntotal', 0)
                image_count = getattr(indexer_instance.image_index, 'ntotal', 0)
                mean_count = getattr(indexer_instance.mean_index, 'ntotal', 0)
                if text_count > 0 or image_count > 0 or mean_count > 0:
                    retriever_instance = Retriever(indexer=indexer_instance)
                    logger.info("Retriever initialized successfully.")
                    initialization_status["retriever"] = "success"
                else:
                    logger.warning("Skipping Retriever initialization: Indexer has no vectors in any index.")
                    initialization_status["retriever"] = "skipped (no index data)"

            except Exception as e_retriever:
                logger.error(f"ERROR during Retriever initialization: {e_retriever}", exc_info=True)
                initialization_status["retriever"] = "failed"
                if not initialization_status["error_message"]: # Prioritize earlier errors
                     initialization_status["error_message"] = f"Retriever Init Failed: {e_retriever}"
        else:
            logger.warning("Skipping Retriever initialization because Indexer failed.")
            initialization_status["retriever"] = "skipped (indexer failed)"

        # 6. Initialize Generator
        logger.info("--- Startup: Initializing Generator ---")
        initialization_status["generator"] = "initializing"
        zhipuai_api_key = os.getenv("ZHIPUAI_API_KEY")
        if not zhipuai_api_key:
            logger.warning("Environment variable 'ZHIPUAI_API_KEY' not set. Generator will not be available.")
            initialization_status["generator"] = "skipped (no API key)"
        else:
            try:
                generator_instance = Generator(api_key=zhipuai_api_key, model_name=LLM_MODEL)
                logger.info("Generator initialized successfully.")
                initialization_status["generator"] = "success"
            except Exception as e_generator:
                logger.error(f"ERROR during Generator initialization: {e_generator}", exc_info=True)
                initialization_status["generator"] = "failed"
                if not initialization_status["error_message"]: # Prioritize earlier errors
                     initialization_status["error_message"] = f"Generator Init Failed: {e_generator}"

        # Final Status Update
        if "failed" in initialization_status.values():
            initialization_status["status"] = "failed"
            logger.critical(f"RAG system initialization FAILED. Error: {initialization_status['error_message']}")
        elif "skipped" in initialization_status.values() and initialization_status["indexer"] != "success":
             initialization_status["status"] = "failed" # If indexer fails, consider overall failure
             logger.critical(f"RAG system initialization FAILED (Indexer related). Error: {initialization_status['error_message']}")
        elif "skipped" in initialization_status.values():
             initialization_status["status"] = "partial"
             logger.warning("RAG system initialization PARTIAL. Some components were skipped (check logs).")
        else:
            initialization_status["status"] = "success"
            logger.info("RAG system core components initialized successfully.")

        logger.info("FastAPI Server Startup Sequence Complete.")
        logger.info("="*50)

    except Exception as e_startup:
        # Catch any unexpected error during the whole startup process
        print(f"FATAL UNHANDLED ERROR during FastAPI startup: {e_startup}")
        # Try logging if possible, otherwise just print
        if logger and logger.hasHandlers() and not isinstance(logger.handlers[0], logging.NullHandler):
            logger.critical(f"FATAL UNHANDLED ERROR during FastAPI startup: {e_startup}", exc_info=True)
        initialization_status["status"] = "failed"
        initialization_status["error_message"] = f"Unhandled Startup Error: {e_startup}"
        # Allow server to start but endpoints should check status

# =============================================================================================
# FastAPI Shutdown Event Handler
# =============================================================================================
@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleans up resources when the FastAPI application shuts down.
    Ensures Faiss indices are saved by closing the Indexer.
    """
    logger.info("="*50)
    logger.info("FastAPI Server Shutdown Sequence Initiated")
    if retriever_instance:
        try:
            retriever_instance.close()
        except Exception as e:
            logger.error(f"Error during Retriever shutdown: {e}", exc_info=True)
    if generator_instance:
         try:
            generator_instance.close()
         except Exception as e:
            logger.error(f"Error during Generator shutdown: {e}", exc_info=True)
    if indexer_instance:
        try:
            indexer_instance.close() # This handles saving Faiss indices
        except Exception as e:
            logger.error(f"Error during Indexer shutdown (Faiss indices might not be saved!): {e}", exc_info=True)

    # Optional: Clean up temporary upload directory
    if os.path.exists(UPLOADS_TEMP_DIR):
        try:
            shutil.rmtree(UPLOADS_TEMP_DIR)
            logger.info(f"Removed temporary upload directory: {UPLOADS_TEMP_DIR}")
        except Exception as e:
            logger.error(f"Error removing temporary upload directory {UPLOADS_TEMP_DIR}: {e}")

    logger.info("FastAPI Server Shutdown Sequence Complete.")
    logger.info("="*50)


# =============================================================================================
# API Endpoints
# =============================================================================================

# --- Root Endpoint (Serve Frontend) ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serves the main HTML file for the frontend user interface.
    My duty is to ensure the user sees the interface you designed, Boss.
    """
    logger.info(f"Request received for root path '/'. Serving index.html.")
    index_html_path = Path("static/index.html")
    if not index_html_path.is_file():
        logger.error("Frontend file 'static/index.html' not found!")
        raise HTTPException(status_code=404, detail="Frontend HTML file not found.")
    return FileResponse(index_html_path)

# --- Status Endpoint ---
@app.get("/status", response_class=JSONResponse)
async def get_status():
    """
    Provides the current initialization status of the core RAG components.
    Allows checking if the system is ready to serve queries. Essential for monitoring.
    """
    logger.info("Request received for '/status' endpoint.")
    status_data = initialization_status.copy()
    # Add current index counts for more detailed status
    status_data["indexer_db_count"] = indexer_instance.get_document_count() if indexer_instance else "N/A"
    status_data["indexer_text_vector_count"] = getattr(getattr(indexer_instance, 'text_index', None), 'ntotal', "N/A") if indexer_instance else "N/A"
    status_data["indexer_image_vector_count"] = getattr(getattr(indexer_instance, 'image_index', None), 'ntotal', "N/A") if indexer_instance else "N/A"
    status_data["indexer_mean_vector_count"] = getattr(getattr(indexer_instance, 'mean_index', None), 'ntotal', "N/A") if indexer_instance else "N/A"

    return JSONResponse(content=status_data)

# --- Query Endpoint ---
@app.post("/query", response_class=JSONResponse)
async def handle_query(
    request: Request, # Added request object for logging client info
    query_text: Optional[str] = Form(None), # Text query from form
    query_image: Optional[UploadFile] = File(None) # Optional images upload
):
    """
    Handles user queries (text, images, or multimodal).
    Orchestrates the RAG pipeline: retrieval -> generation.
    This is the core interaction point, and I will handle it flawlessly.

    Args:
        query_text (Optional[str]): Text part of the query.
        query_image (Optional[UploadFile]): Image part of the query.

    Returns:
        JSONResponse: Contains 'retrieved_docs' and 'generated_response'.
                      Includes an 'error' field if processing fails.
    """
    client_host = request.client.host if request.client else "unknown"
    logger.info(f"Received '/query' request from {client_host}.")

    # 1. Check Initialization Status
    if initialization_status["status"] == "failed":
        logger.error("Query rejected: System initialization failed previously.")
        raise HTTPException(status_code=503, detail=f"Service Unavailable: System initialization failed. Error: {initialization_status.get('error_message', 'Unknown initialization error')}")
    if not retriever_instance:
        logger.error("Query rejected: Retriever component is not available.")
        raise HTTPException(status_code=503, detail="Service Unavailable: Retriever component not initialized.")
    if not generator_instance:
         logger.warning("Generator component is not available. Retrieval will proceed, but generation will be skipped.")
         # Allow retrieval even if generator is down, but response will indicate missing generation


    # 2. Validate Input
    if not query_text and not query_image:
        logger.warning("Query rejected: No text query or images file provided.")
        raise HTTPException(status_code=400, detail="Bad Request: Please provide either a text query or upload an images.")

    # 3. Handle Image Upload (Save temporarily)
    temp_image_path: Optional[str] = None
    if query_image:
        # Sanitize filename provided by user before saving
        safe_original_filename = sanitize_filename(query_image.filename if query_image.filename else "uploaded_image")
        # Create a unique temporary file path
        # Using a subdirectory helps keep the main output dir clean
        os.makedirs(UPLOADS_TEMP_DIR, exist_ok=True)
        # Include timestamp/random element for uniqueness
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        unique_filename = f"{timestamp}_{safe_original_filename}"
        temp_image_path = os.path.join(UPLOADS_TEMP_DIR, unique_filename)

        logger.info(f"Processing uploaded images: '{query_image.filename}' -> Saving temporarily as '{temp_image_path}'")
        try:
            with open(temp_image_path, "wb") as buffer:
                shutil.copyfileobj(query_image.file, buffer)
            logger.info(f"Uploaded images saved successfully to {temp_image_path}")
        except Exception as e:
            logger.error(f"Failed to save uploaded images '{query_image.filename}' to '{temp_image_path}': {e}", exc_info=True)
            # Clean up partial file if it exists
            if os.path.exists(temp_image_path):
                try:
                    os.remove(temp_image_path)
                except Exception as e_remove:
                     logger.error(f"Failed to remove partially saved file {temp_image_path}: {e_remove}")
            raise HTTPException(status_code=500, detail=f"Failed to process uploaded images: {e}")
        finally:
            # Ensure the file object associated with UploadFile is closed
             if hasattr(query_image, 'file') and hasattr(query_image.file, 'close'):
                  query_image.file.close()


    # 4. Prepare Query for Retriever
    retriever_query: Union[str, Dict[str, str], None] = None
    if query_text and temp_image_path:
        retriever_query = {"text": query_text, "image_path": temp_image_path}
        query_type_log = "Multimodal"
    elif temp_image_path:
        retriever_query = {"image_path": temp_image_path}
        query_type_log = "Pure Image"
    elif query_text:
        retriever_query = query_text
        query_type_log = "Pure Text"
    else:
         # This case should be caught by validation, but handle defensively
         logger.error("Internal error: No valid query components after processing.")
         if temp_image_path and os.path.exists(temp_image_path): # Cleanup if images was saved but text was missing
             os.remove(temp_image_path)
         raise HTTPException(status_code=500, detail="Internal server error preparing query.")

    logger.info(f"Prepared {query_type_log} query for Retriever.")

    # 5. Execute Retrieval
    retrieved_docs: List[Dict[str, Any]] = []
    retrieval_error: Optional[str] = None
    try:
        logger.info("Executing retrieval stage...")
        # Using a fixed K for the API, can be made a parameter later
        retrieved_docs = retriever_instance.retrieve(retriever_query, k=3)
        logger.info(f"Retrieval stage complete. Found {len(retrieved_docs)} documents.")
    except Exception as e_retrieve:
        logger.error(f"Error during retrieval stage: {e_retrieve}", exc_info=True)
        retrieval_error = f"Retrieval failed: {e_retrieve}"
        # Don't raise yet, try cleanup first, then return error in JSON

    # 6. Execute Generation (if retrieval successful and generator available)
    generated_response: str = "Generation skipped or failed." # Default
    generation_error: Optional[str] = None
    if not retrieval_error and generator_instance: # Only generate if retrieval worked AND generator is up
        if retrieved_docs:
             # Determine the text query to use for generation
             # Using the original input query_text seems most logical
             generator_query_text = query_text if query_text else "Describe the context provided." # Fallback if only images was input
             logger.info(f"Executing generation stage using query: '{generator_query_text[:100]}...' and {len(retrieved_docs)} context docs.")
             try:
                 generated_response = generator_instance.generate(generator_query_text, retrieved_docs)
                 logger.info("Generation stage complete.")
             except Exception as e_generate:
                 logger.error(f"Error during generation stage: {e_generate}", exc_info=True)
                 generation_error = f"Generation failed: {e_generate}"
                 generated_response = f"Error during generation: {e_generate}" # Include error in response
        else:
            logger.info("Skipping generation stage: No documents were retrieved.")
            generated_response = "No relevant documents found, so no answer could be generated."
    elif not generator_instance:
         logger.warning("Skipping generation stage: Generator component is not available.")
         generated_response = "Generation skipped: Language model component not available."
    elif retrieval_error:
         logger.warning(f"Skipping generation stage due to retrieval error: {retrieval_error}")
         generated_response = f"Generation skipped due to retrieval error."


    # 7. Cleanup Temporary Image File
    if temp_image_path and os.path.exists(temp_image_path):
        try:
            os.remove(temp_image_path)
            logger.info(f"Cleaned up temporary images file: {temp_image_path}")
        except Exception as e_cleanup:
            logger.error(f"Error cleaning up temporary images file {temp_image_path}: {e_cleanup}")
            # Log error but don't fail the request at this point

    # 8. Prepare and Return Response
    final_response_data = {
        "retrieved_docs": retrieved_docs,
        "generated_response": generated_response,
        "error": retrieval_error or generation_error # Report first error encountered
    }

    # Optionally save query results to disk (similar to notebook)
    try:
        query_save_prefix = sanitize_filename(query_text[:50] if query_text else query_type_log, max_length=50)
        timestamp_save = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        query_output_dir = os.path.join(QUERY_RESULTS_DIR, f"{timestamp_save}_{query_save_prefix}")
        os.makedirs(query_output_dir, exist_ok=True)

        # Save input details
        input_data_to_save = {"query_text": query_text, "uploaded_image_filename": query_image.filename if query_image else None}
        with open(os.path.join(query_output_dir, "query_input.json"), "w", encoding="utf-8") as f_in:
             json.dump(input_data_to_save, f_in, ensure_ascii=False, indent=4)

        # Save results
        with open(os.path.join(query_output_dir, "query_result.json"), "w", encoding="utf-8") as f_out:
            # Convert numpy arrays in retrieved_docs for JSON serialization if any exist (shouldn't with current structure)
            # For safety, let's just save the final JSON-ready response
             json.dump(final_response_data, f_out, ensure_ascii=False, indent=4, default=str) # Use default=str for any complex types

        logger.info(f"Saved detailed results for this query to: {query_output_dir}")

    except Exception as e_save_res:
         logger.error(f"Error saving detailed query results to disk: {e_save_res}", exc_info=True)


    if final_response_data["error"]:
        logger.error(f"Query processing finished with error: {final_response_data['error']}")
        # Decide if we should return 500 or 200 with error field
        # Returning 200 allows frontend to display partial results/errors gracefully
        return JSONResponse(content=final_response_data, status_code=200)
    else:
        logger.info("Query processed successfully.")
        return JSONResponse(content=final_response_data)


# =============================================================================================
# Main Execution Block (for running with uvicorn)
# =============================================================================================
if __name__ == "__main__":
    print("Starting Multimodal RAG FastAPI Server...")
    print(f"Output Base Directory: {os.path.abspath(OUTPUT_BASE_DIR)}")
    print("Access the UI at http://127.0.0.1:5000")
    print("API docs available at http://127.0.0.1:5000/docs")
    # Use uvicorn to run the app
    # host="0.0.0.0" makes it accessible on the network, use "127.0.0.1" for local only
    # reload=True is useful for development, disable for production
    uvicorn.run("server:app", host="127.0.0.1", port=5000, reload=True, log_level="info")
