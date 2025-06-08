from argparse import ArgumentParser
parser=ArgumentParser(description="A simple script to demonstrate argument parsing.")
parser.add_argument('--pdf_path', type=str, required=True, help='Path to the PDF file to be processed.')
parser.add_argument('--db_path', type=str, required=False, help='Path to the database file where embeddings will be stored.',default='vecdb.parquet')
parser.add_argument('--embed_model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='Name of the embedding model to use.')   
parser.add_argument('--top_k', type=int, default=3, help='Number of top similar chunks to retrieve.')
parser.add_argument('--additional_filters', type=str, default='{}', help='Additional filters for the search query in JSON format.') 

args = parser.parse_args()

from pathlib import Path
from utils.db_utils import VectorDB, chunk_pdf

def main():
    pdf_path = Path(args.pdf_path)
    db_path = Path(args.db_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"The specified PDF file does not exist: {pdf_path}")
    # Ensure the database file is created if it does not exist
    vector_db = VectorDB(db_path=db_path)
    text_chunks = list(chunk_pdf(pdf_path))
    metadata = [{'source': str(pdf_path), 'page': i} for i in range(len(text_chunks))]
    vector_db.add_chunks(text_chunks=text_chunks, metadata=metadata)

    # Example search query
    query = "RAG"
    additional_filters = eval(args.additional_filters)  # Convert string to dictionary
    
    results = vector_db.search_chunks(query=query, top_k=args.top_k, additional_filters=additional_filters)
    
    for result in results:
        print(result)
if __name__ == "__main__":
    main()