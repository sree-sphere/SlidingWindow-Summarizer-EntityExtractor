import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pdf2image import convert_from_path
import pytesseract
from src.log import logger
from docx2pdf import convert

page_breaker = "$$$$_new"

if os.environ.get("pytesseract_cmd"):
    pytesseract.pytesseract.tesseract_cmd = os.environ.get("pytesseract_cmd")

def docx_to_pdf(docx_path, output_pdf_path):
    convert(docx_path, output_pdf_path)

def pdf_to_images(pdf_path):
    return convert_from_path(pdf_path)

def ocr_images(images):
    extracted_text = []
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image)
        extracted_text.append(text)
    return page_breaker.join(extracted_text)

def docx_to_text_via_ocr(docx_path):
    base_name = os.path.splitext(os.path.basename(docx_path))[0]
    pdf_path = f"{base_name}.pdf"

    print("[1] Converting .docx to .pdf...")
    docx_to_pdf(docx_path, pdf_path)

    print("[2] Converting PDF to images...")
    images = pdf_to_images(pdf_path)

    print(f"[3] Performing OCR on {len(images)} pages...")
    text = ocr_images(images)

    print("[4] Cleaning up temporary files...")
    os.remove(pdf_path)

    return text


def ocr_extract_from_pdf(pdf_path, dpi=300, lang='eng', threads=None, output_file=None):
    """
    Extract text from a PDF using OCR exclusively for all content.
    
    This function converts each page of the PDF to an image and then applies OCR,
    regardless of whether the content is text, tables, or images.
    
    Args:
        pdf_path (str): Path to the PDF file
        dpi (int): DPI for the converted images (higher values give better OCR quality but slower processing)
        lang (str): Language for OCR (default: 'eng')
        threads (int): Number of threads to use for parallel processing (default: None, which uses CPU count)
        output_file (str): Path to save the extracted text (default: None, doesn't save)
        
    Returns:
        str: Extracted text from the PDF
    """
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    logger.info(f"Processing PDF: {pdf_path} with OCR at {dpi} DPI")
    
    # Create a temporary directory for the images
    with tempfile.TemporaryDirectory() as temp_dir:
        # Convert PDF to images
        logger.info("Converting PDF pages to images...")
        try:
            images = convert_from_path(pdf_path, dpi=dpi)
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            raise
        
        logger.info(f"Successfully converted {len(images)} pages to images")
        
        # Process OCR on each image
        all_text = []
        
        # Function to process a single page
        def process_page(args):
            page_num, image = args
            logger.info(f"Performing OCR on page {page_num+1}/{len(images)}...")
            try:
                text = pytesseract.image_to_string(image, lang=lang)
                
                # Save image for debugging if needed
                # image_path = os.path.join(temp_dir, f"page_{page_num+1}.png")
                # image.save(image_path)
                
                return page_num, text
            except Exception as e:
                logger.error(f"Error in OCR for page {page_num+1}: {str(e)}")
                return page_num, f"[OCR ERROR ON PAGE {page_num+1}]"
        
        # Use ThreadPoolExecutor for parallel processing
        max_workers = threads if threads is not None else min(32, os.cpu_count() + 4)
        
        # Only use parallel processing if there are multiple pages
        if len(images) > 1 and max_workers > 1:
            logger.info(f"Processing {len(images)} pages with {max_workers} workers")
            results = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_page = {executor.submit(process_page, (i, img)): i for i, img in enumerate(images)}
                for future in as_completed(future_to_page):
                    page_num, text = future.result()
                    results[page_num] = text
            
            # Ensure pages are in correct order
            all_text = [results[i] for i in range(len(images))]
        else:
            # Process pages sequentially for single page PDFs or if parallel disabled
            all_text = [process_page((i, img))[1] for i, img in enumerate(images)]
        
        # Combine all text
        full_text = page_breaker.join(all_text)
        
        # Save to file if requested
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(full_text)
                logger.info(f"Text saved to {output_file}")
            except Exception as e:
                logger.error(f"Error saving text to file: {str(e)}")
        
        logger.info(f"Successfully extracted {len(full_text)} characters of text")
        return full_text