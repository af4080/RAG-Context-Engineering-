import os
from openai import OpenAI
from pydantic import BaseModel, Field
from rag.chunking import Chunk

MODEL = "openai/gpt-oss-120b"

SYSTEM = """אתה עונה על שאלות של לקוחות חברת הרכב "אופק רכב".
ענה רק על סמך הקטעים שבתוך <chunk>. אל תשתמש בידע כללי.
לכל טענה בתשובה, הוסף ציטוט: מזהה הקטע וציטוט מדויק מתוכו, מילה במילה.
אם הקטעים לא מספיקים כדי לענות, החזר insufficient_context=true, בלי ציטוטים.
ענה בעברית, במשפטים קצרים."""


class Citation(BaseModel):
    chunk_id: str = Field(description="The chunk id, exactly as it appears in <chunk id=...>")
    quote: str = Field(description="An exact quote from that chunk, word for word")


class Answer(BaseModel):
    answer: str
    citations: list[Citation]
    insufficient_context: bool


# אובייקט עזר להתאמת שמות שדות השימוש בטוקנים למה ש-eval_answers.py מצפה (תאימות ל-Anthropic)
class UsageAdapter:
    def __init__(self, openai_usage):
        self.input_tokens = getattr(openai_usage, "prompt_tokens", 0)
        self.output_tokens = getattr(openai_usage, "completion_tokens", 0)
        self.cache_read_input_tokens = 0
        self.cache_creation_input_tokens = 0


def build_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f'<chunk id="{c.id}">\n{c.text}\n</chunk>' for c in chunks)


def ask(question: str, chunks: list[Chunk], client=None) -> tuple[Answer, object]:
    """Returns the parsed Answer and the token usage. Needs GROQ_API_KEY in the environment."""
    if client is None:
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY"),
        )
    
    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{build_context(chunks)}\n\nשאלה: {question}"},
        ],
        response_format=Answer,
        temperature=0.0,
    )
    
    message = response.choices[0].message
    usage = UsageAdapter(response.usage)
    return message.parsed, usage


def verify_citations(answer: Answer, chunks: list[Chunk]) -> list[Citation]:
    """Return the citations that are NOT backed by the chunks.

    A citation is bad if its chunk_id is not one of the chunks we sent,
    or if its quote does not appear in that chunk's text.
    An empty list means every citation checks out.
    """
    chunks_map = {c.id: c.text for c in chunks}
    invalid_citations = []
    
    for citation in answer.citations:
        if citation.chunk_id not in chunks_map:
            invalid_citations.append(citation)
            continue
            
        chunk_text = chunks_map[citation.chunk_id]
        norm_quote = " ".join(citation.quote.split())
        norm_chunk = " ".join(chunk_text.split())
        
        if not norm_quote or norm_quote not in norm_chunk:
            invalid_citations.append(citation.copy())
            
    return invalid_citations