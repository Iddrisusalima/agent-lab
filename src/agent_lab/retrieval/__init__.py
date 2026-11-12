"""Retrieval layer: corpus loading, chunking, the Chroma index and the keyword fallback.

Everything here sits behind the narrow ``Retriever`` protocol so an equivalent
retrieval implementation can be substituted without changing ``search_docs``
(Req 7.7).
"""
