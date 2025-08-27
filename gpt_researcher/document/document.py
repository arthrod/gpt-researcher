import asyncio
import os

from langchain_community.document_loaders import (
    BSHTMLLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredCSVLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)


class DocumentLoader:
    def __init__(self, path: str | list[str]):
        self.path = path

    async def load(self) -> list:
        """
        Load documents from the configured path (a directory path or a list of file paths) using the appropriate file loaders and return extracted page contents.
        
        If self.path is a list, only existing files are queued. If self.path is a string/bytes/os.PathLike it is treated as a directory and scanned recursively. Each file is dispatched to the loader selected by its file extension via _load_document; results are gathered concurrently. Only pages with non-empty `page_content` are included in the result; each returned item is a dict with keys:
        - "raw_content": the page's text content
        - "url": the basename of the page's source metadata
        
        Returns:
            list: A list of dictionaries containing loaded page contents and source basenames.
        
        Raises:
            ValueError: If self.path is not a supported type (str, bytes, os.PathLike, or list thereof), or if no documents could be loaded.
        """
        tasks = []
        if isinstance(self.path, list):
            for file_path in self.path:
                if os.path.isfile(file_path):  # Ensure it's a valid file
                    filename = os.path.basename(file_path)
                    file_name, file_extension_with_dot = os.path.splitext(filename)
                    file_extension = file_extension_with_dot.strip(".").lower()
                    tasks.append(self._load_document(file_path, file_extension))

<<<<<<< HEAD
        elif isinstance(self.path, (str, bytes, os.PathLike)):
            for root, dirs, files in os.walk(self.path):
=======
        elif isinstance(self.path, str | bytes | os.PathLike):
            for root, _dirs, files in os.walk(self.path):
>>>>>>> newdev
                for file in files:
                    file_path = os.path.join(root, file)
                    file_name, file_extension_with_dot = os.path.splitext(file)
                    file_extension = file_extension_with_dot.strip(".").lower()
                    tasks.append(self._load_document(file_path, file_extension))

        else:
            raise ValueError(
                "Invalid type for path. Expected str, bytes, os.PathLike, or list thereof."
            )

        # for root, dirs, files in os.walk(self.path):
        #     for file in files:
        #         file_path = os.path.join(root, file)
        #         file_name, file_extension_with_dot = os.path.splitext(file_path)
        #         file_extension = file_extension_with_dot.strip(".")
        #         tasks.append(self._load_document(file_path, file_extension))

        docs = []
        for pages in await asyncio.gather(*tasks):
            for page in pages:
                if page.page_content:
<<<<<<< HEAD
                    docs.append({
                        "raw_content": page.page_content,
                        "url": os.path.basename(page.metadata['source'])
                    })
=======
                    docs.append(
                        {
                            "raw_content": page.page_content,
                            "url": os.path.basename(page.metadata["source"]),
                        }
                    )
>>>>>>> newdev

        if not docs:
            raise ValueError("🤷 Failed to load any documents!")

        return docs

    async def _load_document(self, file_path: str, file_extension: str) -> list:
        ret_data = []
        try:
            loader_dict = {
                "pdf": PyMuPDFLoader(file_path),
                "txt": TextLoader(file_path),
                "doc": UnstructuredWordDocumentLoader(file_path),
                "docx": UnstructuredWordDocumentLoader(file_path),
                "pptx": UnstructuredPowerPointLoader(file_path),
                "csv": UnstructuredCSVLoader(file_path, mode="elements"),
                "xls": UnstructuredExcelLoader(file_path, mode="elements"),
                "xlsx": UnstructuredExcelLoader(file_path, mode="elements"),
                "md": UnstructuredMarkdownLoader(file_path),
                "html": BSHTMLLoader(file_path),
                "htm": BSHTMLLoader(file_path),
            }

            loader = loader_dict.get(file_extension)
            if loader:
                try:
                    ret_data = loader.load()
                except Exception as e:
                    print(f"Failed to load HTML document : {file_path}")
                    print(e)

        except Exception as e:
            print(f"Failed to load document : {file_path}")
            print(e)

        return ret_data
