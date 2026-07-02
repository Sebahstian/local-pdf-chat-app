"""Test fixtures. The sample PDF is generated on the fly (reportlab, dev-only dep)
so no binary file needs to live in the repo."""

import io

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

PAGE_TEXTS = [
    "Refunds are accepted within 30 days of purchase with a valid receipt. " * 20,
    "After 30 days, purchases can only be exchanged for store credit. " * 20,
    "Contact support at help@example.com for any refund questions. " * 20,
]


@pytest.fixture(scope="session")
def sample_pdf_bytes() -> bytes:
    """A 3-page PDF with enough text per page to produce multiple chunks."""
    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=letter)
    for text in PAGE_TEXTS:
        y = 750
        # naive wrap: one sentence per line, enough for extraction tests
        for line in text.split(". "):
            pdf.drawString(40, y, line[:100])
            y -= 14
        pdf.showPage()
    pdf.save()
    return buf.getvalue()
