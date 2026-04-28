import re
from abc import ABC, abstractmethod


class BaseDiffTool(ABC):

    def _ensure_newline(self, text: str) -> str:
        # Normalize line breaks to Unix style
        text = re.sub(r'\r\n?|\v|\f|\u2028', '\n', text)
        if text and not text.endswith('\n'):
            text += '\n'
        return text
    
    @abstractmethod
    def calculate_diff(self, text1: str, text2: str, **kwargs) -> str:
        """
        Calculate the diff between two texts.
        return: The diff result as a string
        """
    
    @abstractmethod
    def apply_diff(self, text: str, diff: str) -> str:
        """
        Apply a diff to a text.
        
        :param text: The original text
        :param diff: The diff to apply
        :return: The resulting text after applying the diff
        """
