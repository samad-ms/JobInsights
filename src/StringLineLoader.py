from typing import List
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

class StringLineLoader(BaseLoader):
    """
    Custom document loader that reads documents from a string, 
    treating each line as a separate document.
    """

    def __init__(self, data: str) -> None:
        """
        Initializes the loader with the string data.

        Args:
            data: The string containing the documents, each line representing a document.
        """
        self.data = data
        self.documents: List[Document] = []
        self._parse_data()

    def _parse_data(self) -> None:
        """
        Parses the string data into a list of Document objects.
        """
        for line in self.data.splitlines():
            # Create a Document object with page_content set to the line and empty metadata
            document = Document(page_content=line, metadata={})
            self.documents.append(document)

    def load(self) -> List[Document]:
        """
        Returns the list of parsed documents.
        """
        return self.documents

# # Example usage
# data = """
# string
# """
# loader = StringLineLoader(data)
# docs = loader.load()
# docs