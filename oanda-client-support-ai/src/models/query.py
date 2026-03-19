"""
ClientQuery — the input model that flows through the entire agent pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid


@dataclass
class ClientQuery:
    """Represents a single client support query entering the pipeline."""

    text: str
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = "unknown"          # e.g. "chat", "email", "asana-test"
    language: str = "en"
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.text or not self.text.strip():
            raise ValueError("ClientQuery.text must not be empty.")
        self.text = self.text.strip()

    def __repr__(self):
        preview = self.text[:60] + "..." if len(self.text) > 60 else self.text
        return f"ClientQuery(id={self.query_id[:8]}, text='{preview}')"
