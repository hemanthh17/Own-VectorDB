from argparse import ArgumentParser
from utils.db_utils import VectorDB
from pathlib import Path

parser = ArgumentParser(description="A simple script to demonstrate argument parsing.")
parser.add_argument('--operation', type=str, choices=['add', 'search'], required=True, help='Operation to perform on the vector database.')
parser.add_argument('--pdf_path', type=str, required=False, help='Path to the PDF file to be processed.')
parser.add_argument('--db_path', type=str, required=False, help='Path to the database file where embeddings will be stored.', default='vecdb.parquet')
parser.add_argument('--embed_model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='Name of the embedding model to use.')   
parser.add_argument('--top_k', type=int, default=3, help='Number of top similar chunks to retrieve.')
parser.add_argument('--additional_filters', type=str, default=dict(), help='Additional filters for the search query in JSON format.') 
parser.add_argument('--query', type=str, default='', help='Query string for searching in the vector database.')

def main():
    args = parser.parse_args()
    vecdb = VectorDB(Path(args.db_path))
    if args.operation == 'add':
        from utils.db_utils import chunk_pdf
        text_chunks=[]
        for chunk in chunk_pdf(Path(args.pdf_path)):
            text_chunks.append(chunk)
        metadata = [{'page': i + 1, 'source': args.pdf_path, 'text_size': len(chunk)} for i, chunk in enumerate(text_chunks)]
        vecdb.add_chunks(text_chunks, metadata)
    elif args.operation == 'search':
        res=vecdb.search_chunks(query=args.query, top_k=args.top_k, additional_filters=eval(str(args.additional_filters)))
        print(f"Top {args.top_k} results for query '{args.query}':")
        for i, entry in enumerate(res):
            print(f"{i + 1}. Text: {entry}")
    else:
        raise ValueError("Invalid operation specified. Use 'add' or 'search'.")

if __name__ == "__main__":
    main()