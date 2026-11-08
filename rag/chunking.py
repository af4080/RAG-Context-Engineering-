"""Splits documents into chunks. Each chunk keeps its own text and a copy of the document's metadata."""
from dataclasses import dataclass

from rag.loader import Doc


@dataclass
class Chunk:
    id: str             # "<doc id>#<number>", e.g. "of-210-city#2"
    doc_id: str
    text: str           # the original text, exactly as it will be shown to the model
    meta: dict


def chunk_fixed(doc: Doc, size: int = 350, overlap: int = 50) -> list[Chunk]:
    """Baseline: cut every `size` characters, with `overlap` characters repeated between neighbours."""
    chunks = []
    step = size - overlap
    for start in range(0, max(len(doc.text), 1), step):
        piece = doc.text[start:start + size]
        if piece.strip():
            chunks.append(Chunk(f"{doc.id}#{len(chunks)}", doc.id, piece, dict(doc.meta)))
        if start + size >= len(doc.text):
            break
    return chunks


def chunk_by_headings(doc: Doc) -> list[Chunk]:
    """TODO (step 2): one chunk per "## " section.

    - Every chunk starts with a heading path line: "<document title> › <section heading>".
      The text before the first "## " goes in a chunk whose path line is just the title.
    - Skip the "# " title line itself (the path line already carries the title).
    - Keep each section whole, so a table stays together with its header row.
    - Copy the document metadata and add meta["section"] = the section heading.
    - Ids: f"{doc.id}#{n}", numbered from 0.
    """
    chunks = []
    text_content = getattr(doc, 'text', '')
    doc_title = getattr(doc, 'title', '')
    doc_id = getattr(doc, 'id', '')
    base_meta = getattr(doc, 'meta', {})
    if base_meta is None:
        base_meta = {}

    lines = text_content.split('\n')
    
    sections = []
    current_heading = ""
    current_lines = []
    
    for line in lines:
        if line.startswith('# '):
            # דילוג על שורת הכותרת הראשית הראשונה
            continue
        elif line.startswith('## '):
            if current_lines or current_heading != "":
                sections.append((current_heading, current_lines))
            current_heading = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)
            
    if current_lines or current_heading != "":
        sections.append((current_heading, current_lines))
        
    for n, (section_heading, sec_lines) in enumerate(sections):
        # בניית נתיב הכותרת
        if section_heading:
            path_line = f"{doc_title} › {section_heading}"
        else:
            path_line = doc_title
            
        # ניקוי שורות ריקות מיותרות בהתחלה ובסוף
        while sec_lines and sec_lines[0].strip() == '':
            sec_lines.pop(0)
        while sec_lines and sec_lines[-1].strip() == '':
            sec_lines.pop()
            
        sec_text = '\n'.join(sec_lines).strip()
        if path_line and sec_text:
            chunk_text = f"{path_line}\n{sec_text}"
        elif path_line:
            chunk_text = path_line
        else:
            chunk_text = sec_text
            
        meta = dict(base_meta)
        meta["section"] = section_heading
        
        chunk_id = f"{doc_id}#{n}"
        chunks.append(Chunk(id=chunk_id, doc_id=doc_id, text=chunk_text, meta=meta))
        
    return chunks