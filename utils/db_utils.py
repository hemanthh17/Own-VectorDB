import pyarrow 
import pyarrow.parquet as pq
import pdfplumber
import pandas as pd 
from pathlib import Path
import transformers
import torch
import numpy as np
from typing import Generator, List, Dict, Any

def chunk_pdf(filepath: Path) -> Generator[str]:
    with pdfplumber.open(filepath) as f:
        for page in f.pages:
            yield page.extract_text()

def embed_text(text: str, embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    model=transformers.AutoModel.from_pretrained(embed_model_name)
    tokenizer=transformers.AutoTokenizer.from_pretrained(embed_model_name)
    inputs = tokenizer(text, return_tensors='pt', padding= True, truncation= True)
    with torch.no_grad():
        embeddings= model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.numpy().tolist()

class VectorDB:
    def __init__(self,db_path: Path):
        self.db_path = db_path
        self.db: pyarrow.Table = None
        self._load_db()
    def _load_db(self)-> None:
        if self.db_path.exists():
            self.db= pq.read_table(self.db_path)
        else:
            schema = pyarrow.schema([
                ("text", pyarrow.string()),
                ("metadata", pyarrow.struct([
                    ("page", pyarrow.int64()),
                    ("source", pyarrow.string()),
                    ("text_size", pyarrow.int64())
                ])),
                ("embedding", pyarrow.list_(pyarrow.list_(pyarrow.float64())))
            ])
            empty_df = pd.DataFrame(columns=["text", "metadata", "embedding"])
            self.db = pyarrow.Table.from_pandas(empty_df, schema=schema, preserve_index=False)
            pq.write_table(self.db, self.db_path)
    def _add_entry(self,text_chunk:str,metadata: Dict[str, Any]=dict())-> None:
        chunk_entry = {"text": [text_chunk], 'metadata': [metadata], 'embedding': [embed_text(text_chunk)]}
        new_table = pyarrow.Table.from_pandas(pd.DataFrame(chunk_entry))
        self.db = pyarrow.concat_tables([self.db, new_table])
        pq.write_table(self.db, self.db_path)

    def add_chunks(self,text_chunks: List[str],metadata: List[Dict[str,Any]]=[dict()])-> None:
        for chunk,meta in zip(text_chunks,metadata):
            self._add_entry(text_chunk=chunk,metadata={**meta,'text_size':len(chunk)})
    def search_chunks(self, query: str, top_k: int = 5,additional_filters: Dict[str, Any] = dict()) -> List[Dict[str, Any]]:
        query_embedding = embed_text(query)
        # Apply additional_filters if provided
        df = self.db.to_pandas()
        for key, value in additional_filters.items():
            # Check if key is in metadata dict (which is a column of dicts)
            df = df[df['metadata'].apply(lambda m: isinstance(m, dict) and m.get(key) == value)]
        df = self.db.to_pandas()
        for key, value in additional_filters.items():
            if key in df.columns:
                df = df[df[key] == value]
            else:
            # Check inside metadata dict
                df = df[df['metadata'].apply(lambda m: isinstance(m, dict) and m.get(key) == value)]
        if df.empty:
            return []
        db_embeddings = np.array([emb[0] for emb in df['embedding'].tolist()])
        query_embedding = np.array(query_embedding[0])  # shape: (embedding_dim,)
        db_embeddings = np.array([emb[0] for emb in self.db['embedding'].to_pandas().tolist()])  # shape: (n, embedding_dim)
        similarities = np.dot(db_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            entry = self.db.to_pandas().iloc[idx].to_dict()
            entry['similarity'] = similarities[idx]
            results.append(entry['text'])
        return results
    def _delete_entry(self, text_chunk: str) -> None:
        self.db = self.db.filter(~(self.db['text'] == text_chunk))
        pq.write_table(self.db, self.db_path)
    def delete_entries(self, text_chunks: List[str]) -> None:
        for chunk in text_chunks:
            self._delete_entry(chunk)
    def get_all_entries(self) -> List[Dict[str, Any]]:
        return self.db.to_pandas().to_dict(orient='records')
    def clear_db(self) -> None:
        self.db = pyarrow.Table.from_pandas(pd.DataFrame(columns=["text","metadata","embedding"]))
        pq.write_table(self.db, self.db_path)
        

        

    