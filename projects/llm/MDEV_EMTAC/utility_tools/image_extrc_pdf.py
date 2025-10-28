import os
from sqlalchemy.orm import Session
from sqlalchemy import text
import json
from your_image_extraction_module import extract_images_from_pdf  # Import your image extraction function

def add_document_to_db(title, file_path, area="", equipment_group="", model="", asset_number=""):
    try:
        extracted_text = None  # Initialize to None in case of exceptions
        extracted_images_sql = []  # Initialize list to store SQL statements for extracted images
        with Session() as session:
            if file_path.endswith(".pdf"):
                # Extract images from the PDF and obtain SQL statements
                images_sql = extract_images_from_pdf(file_path, "path_to_output_folder")
                extracted_images_sql.extend(images_sql)

                extracted_text = extract_text_from_pdf(file_path)
            elif file_path.endswith(".txt"):
                extracted_text = extract_text_from_txt(file_path)
            else:
                print(f"Unsupported file format: {file_path}")
                return False  # Return False immediately for unsupported formats

            file_name = os.path.basename(file_path)
    
            if extracted_text:
                # Add the document to the CompletedDocument table
                complete_document = CompleteDocument(
                    title=title,
                    area=area,
                    equipment_group=equipment_group,
                    model=model,
                    asset_number=asset_number,
                    file_path=file_name,
                    content=extracted_text
                )
                session.add(complete_document)
                session.commit()
                print(f"Added complete document: {title}")

                # Add the document to the FTS table
                insert_query_fts = "INSERT INTO documents_fts (title, content) VALUES (:title, :content)"
                session.execute(text(insert_query_fts), {"title": title, "content": extracted_text})
                print("Added document to the FTS table.")

                # Split the document into chunks and add them to the Document table
                text_chunks = split_text_into_chunks(extracted_text)
                for i, chunk in enumerate(text_chunks):
                    padded_chunk = ' '.join(split_text_into_chunks(chunk, pad_token="", max_words=150))
                    embedding = generate_embedding(padded_chunk)
                    if embedding is not None:
                        embedding_json = json.dumps(embedding)
                        embedding_bytes = embedding_json.encode('utf-8')
                        document = Document(
                            title=f"{title} - Chunk {i+1}",
                            area=area,
                            equipment_group=equipment_group,
                            model=model,
                            asset_number=asset_number,
                            content=padded_chunk,
                            complete_document_id=complete_document.id,
                            embedding=embedding_bytes
                        )
                        session.add(document)
                        session.commit()
                        print(f"Added chunk {i+1} of document: {title}")
                    else:
                        print(f"Failed to add chunk {i+1} of document: {title}")
                        return False
            else:
                print("No text extracted from the document.")
                return False

            # Execute SQL statements for extracted images
            for sql_statement in extracted_images_sql:
                session.execute(text(sql_statement))
                session.commit()
                print("Added image information to the database.")

            return True  # Indicate success if no errors occurred
    except Exception as e:
        print(f"An error occurred in add_document_to_db: {e}")
        return False
