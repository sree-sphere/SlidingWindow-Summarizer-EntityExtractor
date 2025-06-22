from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from src.extractor import docx_to_text_via_ocr, ocr_extract_from_pdf, page_breaker
from src.summarizer import hierarchical_summarize
from src.log import logger
from src.entity_extraction import hierarchical_extract_entities

app = FastAPI()
# instrument_app(app)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    # Log the validation error
    logger.error(f"Validation error: {exc}")
    # Return a custom error response
    return JSONResponse(
        status_code=422,
        content={"message": "Invalid input format", "detail": exc.errors()},
    )


@app.post("/doc-summarize/")
async def summarize_file(
    document: UploadFile = File(...),
    system_prompt: str = Form(...),
    input_prompt: str = Form(...),
    temperature: float = Form(0.3)
):
    
    # 
    if not all(os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_API_KEY")):
        raise HTTPException(503, "LLM credentials not found! Please check the env variables")
    
    # Save the uploaded file to a temporary location
    suffix = os.path.splitext(document.filename)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await document.read())
        tmp_path = tmp.name

    try:
        # Detect file type and extract text
        if suffix.lower() == ".pdf":
            logger.info("File has a .PDF ext")
            text = ocr_extract_from_pdf(tmp_path)
        elif suffix.lower() in [".docx",".doc"]:
            logger.info("File has a .docx ext")
            text = docx_to_text_via_ocr(tmp_path)
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported file type"})

        # Chunk text for paragraph-wise summarization
        # paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        paragraphs = text.split(page_breaker)
        logger.info(f"Splitted text into {len(paragraphs)} chunks")
        if not paragraphs:
            return JSONResponse(status_code=400, content={"error": "No text found in document"})

        # Run async summarization
        final_summary = await hierarchical_summarize(paragraphs, system_prompt, input_prompt)
        
        
        # Run async entity extraction
        final_entities = await hierarchical_extract_entities(paragraphs, system_prompt, input_prompt, temperature)
        return {"summary": final_summary, "entities": final_entities}

    finally:
        os.remove(tmp_path)
