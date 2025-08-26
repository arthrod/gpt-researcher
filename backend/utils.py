import logging
import os
import traceback
import urllib

import aiofiles
import mistune

# Configure logger
logger = logging.getLogger(__name__)


async def write_to_file(filename: str, text: str) -> None:
    """Asynchronously write text to a file in UTF-8 encoding.

    Args:
        filename (str): The filename to write to.
        text (str): The text to write.

    Raises:
        ValueError: When filename or text is invalid.
        IOError: When file operations fail.
        PermissionError: When lacking write permissions.
    """
    if not filename or not filename.strip():
        raise ValueError("Filename cannot be empty or whitespace only")

    if not isinstance(text, str):
        logger.warning(
            f"Non-string input provided (type: {type(text).__name__}), converting to string"
        )
        text = str(text)

    # Ensure directory exists
    directory = os.path.dirname(filename)
    if directory:
        try:
            os.makedirs(directory, exist_ok=True)
        except PermissionError as e:
            logger.error(f"Permission denied creating directory {directory}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            raise

    try:
        # Convert text to UTF-8, replacing any problematic characters
        text_utf8 = text.encode("utf-8", errors="replace").decode("utf-8")

        async with aiofiles.open(filename, "w", encoding="utf-8") as file:
            await file.write(text_utf8)

        logger.debug(f"Successfully wrote {len(text_utf8)} characters to {filename}")

    except PermissionError as e:
        logger.error(f"Permission denied writing to {filename}: {e}")
        raise
    except OSError as e:
        logger.error(f"OS error writing to {filename}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error writing to {filename}: {e}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        raise


async def write_text_to_md(text: str, filename: str = "") -> str:
    """Writes text to a Markdown file and returns the file path.

    Args:
        text (str): Text to write to the Markdown file.

    Returns:
        str: The file path of the generated Markdown file.
    """
    file_path = f"outputs/{filename[:60]}.md"
    await write_to_file(file_path, text)
    return urllib.parse.quote(file_path)


async def write_md_to_pdf(text: str, filename: str = "") -> str:
    """Converts Markdown text to a PDF file and returns the file path.

    Args:
        text (str): Markdown text to convert.
        filename (str): Base filename for the output file.

    Returns:
        str: The encoded file path of the generated PDF, or empty string on error.

    Raises:
        ImportError: When required dependencies are not available.
        IOError: When file operations fail.
    """
    if not text or not text.strip():
        logger.warning("Empty or whitespace-only text provided for PDF conversion")
        return ""

    file_path = f"outputs/{filename[:60]}.pdf"

    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    try:
        from md2pdf.core import md2pdf

        # Validate markdown content before processing
        if len(text) > 1000000:  # 1MB limit
            logger.warning(
                f"Text is very large ({len(text)} chars), truncating for PDF conversion"
            )
            text = text[:1000000] + "\n\n[Content truncated due to size limits]"

        md2pdf(
            file_path,
            md_content=text,
            css_file_path="./frontend/pdf_styles.css",
            base_url=None,
        )

        logger.info(f"PDF report successfully written to {file_path}")

    except ImportError as e:
        logger.error(f"Missing required dependencies for PDF conversion: {e}")
        logger.error("Please install: pip install md2pdf")
        return ""
    except FileNotFoundError as e:
        logger.error(f"CSS file not found for PDF styling: {e}")
        logger.warning("Attempting PDF conversion without custom styling")
        try:
            md2pdf(file_path, md_content=text)
            logger.info(f"PDF report written to {file_path} (without custom styling)")
        except Exception as fallback_e:
            logger.error(f"Fallback PDF conversion also failed: {fallback_e}")
            return ""
    except PermissionError as e:
        logger.error(f"Permission denied when writing PDF file: {e}")
        logger.error(
            f"Check write permissions for directory: {os.path.dirname(file_path)}"
        )
        return ""
    except Exception as e:
        logger.error(f"Unexpected error in PDF conversion: {e}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return ""

    try:
        encoded_file_path = urllib.parse.quote(file_path)
        return encoded_file_path
    except Exception as e:
        logger.error(f"Error encoding file path: {e}")
        return file_path  # Return unencoded path as fallback


async def write_md_to_word(text: str, filename: str = "") -> str:
    """Converts Markdown text to a DOCX file and returns the file path.

    Args:
        text (str): Markdown text to convert.
        filename (str): Base filename for the output file.

    Returns:
        str: The encoded file path of the generated DOCX, or empty string on error.

    Raises:
        ImportError: When required dependencies are not available.
        IOError: When file operations fail.
    """
    if not text or not text.strip():
        logger.warning("Empty or whitespace-only text provided for DOCX conversion")
        return ""

    file_path = f"outputs/{filename[:60]}.docx"

    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    try:
        from docx import Document
        from htmldocx import HtmlToDocx

        # Validate markdown content before processing
        if len(text) > 1000000:  # 1MB limit
            logger.warning(
                f"Text is very large ({len(text)} chars), truncating for DOCX conversion"
            )
            text = text[:1000000] + "\n\n[Content truncated due to size limits]"

        # Convert report markdown to HTML
        try:
            html = mistune.html(text)
        except Exception as e:
            logger.error(f"Error converting markdown to HTML: {e}")
            # Fallback: treat as plain text
            html = f"<html><body><pre>{text}</pre></body></html>"

        # Create a document object
        doc = Document()

        # Convert the html generated from the report to document format
        HtmlToDocx().add_html_to_document(html, doc)

        # Saving the docx document to file_path
        doc.save(file_path)

        logger.info(f"DOCX report successfully written to {file_path}")

    except ImportError as e:
        logger.error(f"Missing required dependencies for DOCX conversion: {e}")
        logger.error("Please install: pip install python-docx htmldocx")
        return ""
    except PermissionError as e:
        logger.error(f"Permission denied when writing DOCX file: {e}")
        logger.error(
            f"Check write permissions for directory: {os.path.dirname(file_path)}"
        )
        return ""
    except Exception as e:
        logger.error(f"Unexpected error in DOCX conversion: {e}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return ""

    try:
        encoded_file_path = urllib.parse.quote(file_path)
        return encoded_file_path
    except Exception as e:
        logger.error(f"Error encoding file path: {e}")
        return file_path  # Return unencoded path as fallback
