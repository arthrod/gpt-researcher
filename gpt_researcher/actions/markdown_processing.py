import re

import markdown


def extract_headers(markdown_text: str) -> list[dict]:
    """
    Extract headers from markdown text.

    Args:
        markdown_text (str): The markdown text to process.

    Returns:
        List[Dict]: A list of dictionaries representing the header structure.
    """
    headers = []
    parsed_md = markdown.markdown(markdown_text)
    lines = parsed_md.split("\n")

    stack = []
    for line in lines:
        if line.startswith("<h") and len(line) > 2 and line[2].isdigit():
            level = int(line[2])
            header_text = line[line.index(">") + 1 : line.rindex("<")]

            while stack and stack[-1]["level"] >= level:
                stack.pop()

            header = {
                "level": level,
                "text": header_text,
            }
            if stack:
                stack[-1].setdefault("children", []).append(header)
            else:
                headers.append(header)

            stack.append(header)

    return headers


def extract_sections(markdown_text: str) -> list[dict[str, str]]:
    """
    Return a list of written sections found in the given Markdown text.
    
    Converts the Markdown to HTML, then finds each header (<h1>–<h6>) and the HTML content that follows it until the next header or end of document. HTML tags are stripped from the content and only sections with non-empty text are included.
    
    Parameters:
        markdown_text (str): Markdown source to extract sections from.
    
    Returns:
        List[Dict[str, str]]: A list of mappings with keys:
            - "section_title": header text (str)
            - "written_content": section body with HTML tags removed and trimmed (str)
    """
    sections = []
    parsed_md = markdown.markdown(markdown_text)

<<<<<<< HEAD
    pattern = r"<h\d>(.*?)</h\d>(.*?)(?=<h\d>|$)"
=======
    pattern = r'<h\d>(.*?)</h\d>(.*?)(?=<h\d>|$)'
>>>>>>> 1027e1d0 (Fix linting issues)
    matches = re.findall(pattern, parsed_md, re.DOTALL)

    for title, content in matches:
        clean_content = re.sub(r"<.*?>", "", content).strip()
        if clean_content:
<<<<<<< HEAD
            sections.append(
                {"section_title": title.strip(), "written_content": clean_content}
            )
=======
            sections.append({
                "section_title": title.strip(),
                "written_content": clean_content
            })
>>>>>>> 1027e1d0 (Fix linting issues)

    return sections


def table_of_contents(markdown_text: str) -> str:
    """
    Build a nested Markdown table of contents from the headings in the given Markdown text.
    
    The function extracts headings (via extract_headers) and returns a Markdown string starting with
    "## Table of Contents" followed by a bullet list of headers. Nested headers are represented by
    indentation of 4 spaces per level. If an error occurs during TOC generation, the original
    markdown_text is returned unchanged.
    Parameters:
        markdown_text (str): Markdown source to analyze for headings.
    
    Returns:
        str: Markdown containing the generated table of contents, or the original markdown_text on error.
    """

    def generate_table_of_contents(headers, indent_level=0):
        toc = ""
        for header in headers:
            toc += " " * (indent_level * 4) + "- " + header["text"] + "\n"
            if "children" in header:
                toc += generate_table_of_contents(header["children"], indent_level + 1)
        return toc

    try:
        headers = extract_headers(markdown_text)
        toc = "## Table of Contents\n\n" + generate_table_of_contents(headers)
        return toc
    except Exception as e:
        print("table_of_contents Exception : ", e)
        return markdown_text


def add_references(report_markdown: str, visited_urls: set) -> str:
    """
    Add references to the markdown report.

    Args:
        report_markdown (str): The existing markdown report.
        visited_urls (set): A set of URLs that have been visited during research.

    Returns:
        str: The updated markdown report with added references.
    """
    try:
        url_markdown = "\n\n\n## References\n\n"
        url_markdown += "".join(f"- [{url}]({url})\n" for url in visited_urls)
        updated_markdown_report = report_markdown + url_markdown
        return updated_markdown_report
    except Exception as e:
        print(f"Encountered exception in adding source urls : {e}")
        return report_markdown
