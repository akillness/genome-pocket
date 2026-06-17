"""Text splitting operations for CocoIndex."""
from typing import List
from cocoindex.resources.chunk import Chunk, Position

class RecursiveSplitter:
    def split(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Chunk]:
        # A simple recursive/character-based splitter for markdown/text
        chunks = []
        start = 0
        text_len = len(text)
        
        if text_len == 0:
            return []
            
        while start < text_len:
            end = min(start + chunk_size, text_len)
            # Try to find a nice boundary (newline or space) if we are not at the end
            if end < text_len:
                last_newline = text.rfind("\n", start, end)
                if last_newline != -1 and last_newline > start + chunk_size // 2:
                    end = last_newline + 1
                else:
                    last_space = text.rfind(" ", start, end)
                    if last_space != -1 and last_space > start + chunk_size // 2:
                        end = last_space + 1
            
            chunk_text = text[start:end]
            chunks.append(Chunk(
                text=chunk_text,
                start=Position(char_offset=start),
                end=Position(char_offset=end)
            ))
            
            # Advance start by chunk_size - overlap
            step = chunk_size - chunk_overlap
            if step <= 0:
                step = chunk_size
            start += step
            if start >= text_len or end == text_len:
                break
                
        return chunks
