from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import debug_id, info_id, error_id, warning_id, get_request_id, logger
from modules.emtacdb.emtacdb_fts import Part, Image, PartsPositionImageAssociation, Drawing, DrawingPartAssociation
from sqlalchemy import and_, text, or_
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
import psycopg2
from psycopg2.extras import execute_values
import os
import sys
import uuid
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Union
import json

# Import logging and database configurations
from modules.configuration.log_config import (
    logger, debug_id, info_id, warning_id, error_id,
    set_request_id, get_request_id, log_timed_operation,
    with_request_id
)
from modules.configuration.config_env import DatabaseConfig
import subprocess
import time
import psutil
from dotenv import load_dotenv



@dataclass
class ImagePosition:
    """Represents an image's position within the document."""
    page_number: int
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    image_index: int
    estimated_size: Tuple[int, int]  # width, height
    content_type: str  # 'figure', 'diagram', 'photo', etc.

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'page_number': self.page_number,
            'bbox': self.bbox,
            'image_index': self.image_index,
            'estimated_size': self.estimated_size,
            'content_type': self.content_type
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImagePosition':
        """Create from dictionary."""
        return cls(
            page_number=data['page_number'],
            bbox=tuple(data['bbox']),
            image_index=data['image_index'],
            estimated_size=tuple(data['estimated_size']),
            content_type=data['content_type']
        )


@dataclass
class ChunkBoundary:
    """Represents where a chunk should be split."""
    page_number: int
    start_position: float  # Y coordinate on page
    end_position: float
    chunk_type: str  # 'text', 'heading', 'caption', etc.
    associated_images: List[int]  # Indices of related images
    context_data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'page_number': self.page_number,
            'start_position': self.start_position,
            'end_position': self.end_position,
            'chunk_type': self.chunk_type,
            'associated_images': self.associated_images,
            'context_data': self.context_data
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChunkBoundary':
        """Create from dictionary."""
        return cls(
            page_number=data['page_number'],
            start_position=data['start_position'],
            end_position=data['end_position'],
            chunk_type=data['chunk_type'],
            associated_images=data['associated_images'],
            context_data=data['context_data']
        )


@dataclass
class DocumentStructureMap:
    """Complete mapping of document structure."""
    total_pages: int
    image_positions: List[ImagePosition]
    chunk_boundaries: List[ChunkBoundary]
    page_layouts: Dict[int, Dict[str, Any]]
    extraction_plan: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'total_pages': self.total_pages,
            'image_positions': [img.to_dict() for img in self.image_positions],
            'chunk_boundaries': [chunk.to_dict() for chunk in self.chunk_boundaries],
            'page_layouts': self.page_layouts,
            'extraction_plan': self.extraction_plan,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentStructureMap':
        """Create from dictionary."""
        return cls(
            total_pages=data['total_pages'],
            image_positions=[ImagePosition.from_dict(img) for img in data['image_positions']],
            chunk_boundaries=[ChunkBoundary.from_dict(chunk) for chunk in data['chunk_boundaries']],
            page_layouts=data['page_layouts'],
            extraction_plan=data['extraction_plan'],
            metadata=data['metadata']
        )

    def save_to_file(self, file_path: str) -> None:
        """Save structure map to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, file_path: str) -> 'DocumentStructureMap':
        """Load structure map from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

class PostgreSQLDatabaseManager:
    """Enhanced base class for PostgreSQL database management operations with modern patterns."""

    def __init__(self, session=None, request_id=None):
        self.session_provided = session is not None
        self.db_config = DatabaseConfig()
        self.session = session or self.db_config.get_main_session()
        self.request_id = request_id or get_request_id()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.session_provided:
            self.session.close()
            debug_id("Closed PostgreSQL database session", self.request_id)

    @contextmanager
    def transaction(self):
        """Enhanced context manager for database transactions with proper rollback."""
        try:
            yield self.session
            self.session.commit()
            debug_id("PostgreSQL transaction committed successfully", self.request_id)
        except Exception as e:
            self.session.rollback()
            error_id(f"PostgreSQL transaction failed, rolled back: {str(e)}", self.request_id, exc_info=True)
            raise

    @contextmanager
    def savepoint(self):
        """Context manager for PostgreSQL savepoints."""
        savepoint = self.session.begin_nested()
        try:
            yield self.session
            savepoint.commit()
            debug_id("PostgreSQL savepoint committed", self.request_id)
        except Exception as e:
            savepoint.rollback()
            debug_id(f"PostgreSQL savepoint rolled back: {e}", self.request_id)
            raise

    def commit(self):
        """Commit the current transaction."""
        try:
            self.session.commit()
            debug_id("PostgreSQL transaction committed", self.request_id)
        except Exception as e:
            self.session.rollback()
            error_id(f"PostgreSQL transaction failed, rolled back: {str(e)}", self.request_id, exc_info=True)
            raise

    def commit_with_retry(self, max_retries=3, backoff_factor=0.5):
        """Commit the current transaction with retry logic for transient errors."""
        attempt = 0
        while attempt < max_retries:
            try:
                self.session.commit()
                debug_id(f"PostgreSQL transaction committed in {attempt} attempts", self.request_id)
                return True
            except SQLAlchemyError as e:
                attempt += 1
                if attempt == max_retries:
                    self.session.rollback()
                    error_id(f"PostgreSQL transaction failed after {max_retries} attempts: {str(e)}", self.request_id,
                             exc_info=True)
                    return False
                sleep_time = backoff_factor * (2 ** (attempt - 1))
                error_id(
                    f"PostgreSQL commit failed (attempt {attempt}/{max_retries}): {e}. Retrying after {sleep_time}s",
                    self.request_id)
                time.sleep(sleep_time)
        return False

    def execute_raw_sql(self, sql, params=None):
        """Execute raw SQL with optional parameters and enhanced error handling."""
        try:
            result = self.session.execute(text(sql), params or {})
            debug_id("PostgreSQL raw SQL executed successfully", self.request_id)
            return result
        except Exception as e:
            error_id(f"Error executing PostgreSQL raw SQL: {str(e)}", self.request_id, exc_info=True)
            raise

    def bulk_insert(self, table_name, data, columns):
        """Enhanced bulk insert using PostgreSQL-specific optimizations."""
        try:
            if not data:
                warning_id("No data provided for bulk insert", self.request_id)
                return

            # Set PostgreSQL-specific optimizations
            with self.savepoint():
                self.session.execute(text("SET work_mem = '4MB'"))
                self.session.execute(text("SET maintenance_work_mem = '128MB'"))

            # Get the raw connection
            connection = self.session.connection().connection
            cursor = connection.cursor()

            # Prepare the SQL
            cols = ', '.join(f'"{col}"' for col in columns)
            sql = f'INSERT INTO "{table_name}" ({cols}) VALUES %s'

            # Use execute_values for efficient bulk insert
            execute_values(cursor, sql, data, page_size=1000)

            info_id(f"Bulk inserted {len(data)} rows into {table_name}", self.request_id)

            # Analyze table after bulk insert for better query planning
            self._analyze_table(table_name)

        except Exception as e:
            error_id(f"Error in PostgreSQL bulk insert: {str(e)}", self.request_id, exc_info=True)
            raise

    def _analyze_table(self, table_name):
        """Analyze table for better query performance."""
        try:
            with self.savepoint():
                self.session.execute(text(f'ANALYZE "{table_name}"'))
            debug_id(f"Analyzed PostgreSQL table {table_name}", self.request_id)
        except Exception as e:
            debug_id(f"Table analysis skipped for {table_name}: {e}", self.request_id)

class PostgreSQLDocumentStructureManager(PostgreSQLDatabaseManager):
    def __init__(self, session=None, request_id=None):
        super().__init__(session, request_id)

    @with_request_id
    def analyze_document_structure(self, file_path: str, request_id=None, ocr_content: List[str] = None) -> DocumentStructureMap:
        info_id(f"Starting PostgreSQL-backed document structure analysis: {file_path}", request_id)

        try:
            doc = fitz.open(file_path)
            structure_map = DocumentStructureMap(
                total_pages=len(doc),
                image_positions=[],
                chunk_boundaries=[],
                page_layouts={},
                extraction_plan={},
                metadata={
                    'analyzed_at': datetime.now().isoformat(),
                    'file_path': file_path,
                    'analysis_version': '1.0',
                    'analyzer': 'PostgreSQLDocumentStructureManager'
                }
            )

            info_id(f"Analyzing {structure_map.total_pages} pages with PostgreSQL backend", request_id)

            for page_num in range(len(doc)):
                page = doc[page_num]
                page_ocr = ocr_content[page_num] if ocr_content and page_num < len(ocr_content) else ""
                page_analysis = self._analyze_page_structure(page, page_num, request_id, page_ocr)

                structure_map.page_layouts[page_num] = page_analysis['layout']
                structure_map.image_positions.extend(page_analysis['images'])
                structure_map.chunk_boundaries.extend(page_analysis['chunks'])

            structure_map.extraction_plan = self._create_extraction_plan(structure_map, request_id)
            self._store_structure_analysis(structure_map, request_id)
            doc.close()

            info_id(f"PostgreSQL structure analysis complete: {len(structure_map.image_positions)} images, "
                    f"{len(structure_map.chunk_boundaries)} chunk boundaries", request_id)
            return structure_map

        except Exception as e:
            error_id(f"Error in PostgreSQL document structure analysis: {e}", request_id, exc_info=True)
            raise

    def _analyze_page_structure(self, page, page_num: int, request_id=None, ocr_content: str = None):
        debug_id(f"Analyzing page {page_num} structure", request_id)

        try:
            page_analysis = {
                'layout': {
                    'page_number': page_num,
                    'page_size': page.rect,
                    'rotation': page.rotation,
                    'text_blocks': [],
                    'image_blocks': [],
                    'layout_type': 'unknown'
                },
                'images': [],
                'chunks': []
            }

            # Get standard images
            image_list = page.get_images(full=True)
            debug_id(f"Page {page_num}: Found {len(image_list)} standard images", request_id)
            for img_index, img in enumerate(image_list):
                try:
                    img_rects = page.get_image_rects(img[0])
                    img_rect = img_rects[0] if img_rects else fitz.Rect(0, 0, 100, 100)
                    content_type = 'image/png'
                    debug_id(f"Page {page_num}, Image {img_index}: Rect {img_rect}, Xref {img[0]}, Type {content_type}", request_id)

                    image_pos = ImagePosition(
                        page_number=page_num,
                        bbox=(img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1),
                        image_index=img_index,
                        estimated_size=(int(img_rect.width), int(img_rect.height)),
                        content_type=content_type
                    )
                    page_analysis['images'].append(image_pos)
                    page_analysis['layout']['image_blocks'].append({
                        'bbox': (img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1),
                        'size': (int(img_rect.width), int(img_rect.height)),
                        'type': content_type
                    })
                except Exception as e:
                    debug_id(f"Error analyzing standard image {img_index} on page {page_num}: {e}", request_id)
                    continue

            # Parse SVG images from OCR content
            svg_count = 0
            if ocr_content:
                svg_images = re.findall(r'<img class="imgSvg" id = "([^"]+)" src="data:image/svg\+xml;base64,([^"]+)"', ocr_content)
                debug_id(f"Page {page_num}: Found {len(svg_images)} SVG images in OCR content", request_id)
                for svg_index, (svg_id, svg_data) in enumerate(svg_images):
                    try:
                        img_rect = fitz.Rect(50, 50, 150, 150)
                        image_index = len(image_list) + svg_count + svg_index
                        image_pos = ImagePosition(
                            page_number=page_num,
                            bbox=(img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1),
                            image_index=image_index,
                            estimated_size=(int(img_rect.width), int(img_rect.height)),
                            content_type='image/svg+xml',
                            metadata={'svg_id': svg_id, 'svg_data': svg_data}
                        )
                        page_analysis['images'].append(image_pos)
                        page_analysis['layout']['image_blocks'].append({
                            'bbox': (img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1),
                            'size': (int(img_rect.width), int(img_rect.height)),
                            'type': 'image/svg+xml',
                            'svg_id': svg_id
                        })
                        svg_count += 1
                        debug_id(f"Page {page_num}, SVG Image {svg_id}: Rect {img_rect}", request_id)
                    except Exception as e:
                        debug_id(f"Error analyzing SVG image {svg_id} on page {page_num}: {e}", request_id)
                        continue

            # Fallback: Vector graphics
            if not svg_count:
                drawings = page.get_drawings()
                for drawing in drawings:
                    if drawing['type'] in ['f', 's']:
                        try:
                            img_rect = drawing['rect']
                            image_index = len(image_list) + svg_count
                            image_id = f"svg_{page_num}_{image_index}"
                            content_type = 'image/svg+xml'
                            debug_id(f"Page {page_num}, SVG {image_index}: Rect {img_rect}, ID {image_id}", request_id)

                            image_pos = ImagePosition(
                                page_number=page_num,
                                bbox=(img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1),
                                image_index=image_index,
                                estimated_size=(int(img_rect.width), int(img_rect.height)),
                                content_type=content_type
                            )
                            page_analysis['images'].append(image_pos)
                            page_analysis['layout']['image_blocks'].append({
                                'bbox': (img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1),
                                'size': (int(img_rect.width), int(img_rect.height)),
                                'type': content_type,
                                'svg_id': image_id
                            })
                            svg_count += 1
                        except Exception as e:
                            debug_id(f"Error analyzing drawing {svg_count} on page {page_num}: {e}", request_id)
                            continue
                debug_id(f"Page {page_num}: Found {svg_count} vector/SVG drawings", request_id)

            text_blocks = page.get_text("dict")
            self._analyze_text_layout(text_blocks, page_analysis, page_num, request_id)
            page_analysis['layout']['layout_type'] = self._determine_layout_type(page_analysis['layout'])
            page_analysis['chunks'] = self._create_page_chunk_boundaries(page_analysis, page_num, request_id)

            debug_id(f"Page {page_num}: Created {len(page_analysis['chunks'])} chunk boundaries", request_id)
            return page_analysis

        except Exception as e:
            error_id(f"Error analyzing page {page_num}: {e}", request_id)
            return {'layout': {}, 'images': [], 'chunks': []}

    def _extract_images_with_guidance(self, file_path: str, complete_document_id: int,
                                     structure_map, session, request_id=None) -> int:
        try:
            doc = fitz.open(file_path)
            images_extracted = 0
            image_positions = structure_map.image_positions
            debug_id(f"Extracting {len(image_positions)} images for document {complete_document_id}", request_id)

            os.makedirs("DB_IMAGES", exist_ok=True)

            for image_data in image_positions:
                try:
                    page_num = image_data.page_number
                    image_index = image_data.image_index
                    image_id = f"img_{page_num}_{image_index}"
                    content_type = image_data.content_type
                    page = doc[page_num]

                    if content_type == 'image/svg+xml':
                        file_path = f"DB_IMAGES/{image_id}.svg"
                        svg_data = image_data.metadata.get('svg_data') if hasattr(image_data, 'metadata') else None
                        if svg_data:
                            with open(file_path, 'wb') as f:
                                f.write(base64.b64decode(svg_data))
                        else:
                            with open(file_path, 'w') as f:
                                f.write("<!-- Placeholder SVG -->")
                        description = f"SVG image from page {page_num}"
                    else:
                        image_list = page.get_images(full=True)
                        if image_index < len(image_list):
                            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                            file_path = f"DB_IMAGES/{image_id}.png"
                            pix.save(file_path)
                            description = f"Extracted image from page {page_num}"
                        else:
                            debug_id(f"Image index {image_index} not found on page {page_num}", request_id)
                            continue

                    image_record = Image(
                        title=f"{image_id}_{content_type.replace('/', '_')}",
                        description=description,
                        file_path=file_path
                    )
                    session.add(image_record)
                    session.flush()

                    association = ImageCompletedDocumentAssociation(
                        image_id=image_record.id,
                        complete_document_id=complete_document_id,
                        page_number=page_num,
                        association_method='structure_guided',
                        confidence_score=0.9,
                        context_metadata=json.dumps({
                            'image_id': image_id,
                            'content_type': content_type,
                            'page_number': page_num,
                            'bbox': image_data.bbox,
                            'estimated_size': image_data.estimated_size,
                            'extracted_at': datetime.now().isoformat(),
                            'request_id': request_id
                        })
                    )
                    session.add(association)
                    images_extracted += 1

                except Exception as e:
                    error_id(f"Error extracting image {image_id}: {e}", request_id)
                    continue

            session.commit()
            doc.close()
            debug_id(f"Extracted {images_extracted} images with structure guidance", request_id)
            return images_extracted

        except Exception as e:
            session.rollback()
            error_id(f"Error in guided image extraction: {e}", request_id)
            return 0

    def _store_structure_analysis(self, structure_map: DocumentStructureMap, request_id=None):
        try:
            with self.transaction():
                self.execute_raw_sql("""
                    CREATE TABLE IF NOT EXISTS document_structure_analysis (
                        id SERIAL PRIMARY KEY,
                        file_path TEXT NOT NULL,
                        analysis_date TIMESTAMP DEFAULT NOW(),
                        total_pages INTEGER,
                        total_images INTEGER,
                        total_chunks INTEGER,
                        structure_data JSONB,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)

                self.execute_raw_sql("""
                    INSERT INTO document_structure_analysis 
                    (file_path, total_pages, total_images, total_chunks, structure_data, metadata)
                    VALUES (:file_path, :total_pages, :total_images, :total_chunks, :structure_data, :metadata)
                """, {
                    'file_path': structure_map.metadata.get('file_path'),
                    'total_pages': structure_map.total_pages,
                    'total_images': len(structure_map.image_positions),
                    'total_chunks': len(structure_map.chunk_boundaries),
                    'structure_data': json.dumps(structure_map.to_dict()),
                    'metadata': json.dumps(structure_map.metadata)
                })

                debug_id("Stored structure analysis in PostgreSQL", request_id)

        except Exception as e:
            warning_id(f"Could not store structure analysis: {e}", request_id)

    def get_stored_structure_analysis(self, file_path: str, request_id=None) -> Optional[DocumentStructureMap]:
        try:
            result = self.execute_raw_sql("""
                SELECT structure_data FROM document_structure_analysis 
                WHERE file_path = :file_path 
                ORDER BY analysis_date DESC 
                LIMIT 1
            """, {'file_path': file_path}).fetchone()

            if result:
                structure_data = json.loads(result[0])
                return DocumentStructureMap.from_dict(structure_data)

            return None

        except Exception as e:
            debug_id(f"Could not retrieve stored structure analysis: {e}", request_id)
            return None

    @with_request_id
    def guided_extraction_with_postgresql(self, file_path: str, metadata: Dict[str, Any],
                                          request_id=None) -> Tuple[bool, Dict[str, Any], int]:
        info_id(f"Starting PostgreSQL-guided extraction: {file_path}", request_id)

        try:
            structure_map = self.get_stored_structure_analysis(file_path, request_id)

            if not structure_map:
                structure_map = self.analyze_document_structure(file_path, request_id, ocr_content=None)
            else:
                info_id("Using cached structure analysis from PostgreSQL", request_id)

            extraction_result = self._perform_postgresql_guided_extraction(
                file_path, structure_map, metadata, request_id
            )

            if not extraction_result['success']:
                return False, extraction_result, 500

            association_result = self._create_postgresql_associations(
                extraction_result['complete_document_id'],
                structure_map.extraction_plan,
                request_id
            )

            final_result = {
                'success': True,
                'complete_document_id': extraction_result['complete_document_id'],
                'chunks_created': extraction_result['chunks_created'],
                'images_extracted': extraction_result['images_extracted'],
                'associations_created': association_result['associations_created'],
                'structure_analysis': {
                    'total_pages_analyzed': structure_map.total_pages,
                    'chunks_planned': len(structure_map.chunk_boundaries),
                    'images_planned': len(structure_map.image_positions),
                    'cached_analysis_used': structure_map is not None
                },
                'processing_method': 'postgresql_structure_guided'
            }

            info_id(f"PostgreSQL guided extraction completed: {final_result}", request_id)
            return True, final_result, 200

        except Exception as e:
            error_id(f"Error in PostgreSQL guided extraction: {e}", request_id, exc_info=True)
            return False, {'error': str(e), 'success': False}, 500

    def _create_postgresql_associations(self, complete_document_id: int,
                                        extraction_plan: Dict[str, Any],
                                        request_id=None) -> Dict[str, Any]:
        try:
            associations_created = 0

            with self.transaction():
                chunks = self.session.query(Document).filter(
                    Document.complete_document_id == complete_document_id
                ).all()

                images = self.session.query(Image).filter(
                    Image.complete_document_id == complete_document_id
                ).all()

                chunk_mapping = {}
                image_mapping = {}

                for chunk in chunks:
                    chunk_metadata = json.loads(chunk.metadata) if chunk.metadata else {}
                    if chunk_metadata.get('structure_guided'):
                        page_num = chunk_metadata.get('page_number')
                        chunk_type = chunk_metadata.get('chunk_type')
                        key = f"chunk_{page_num}_{chunk_type}"
                        chunk_mapping[key] = chunk.id

                for image in images:
                    image_metadata = json.loads(image.metadata) if image.metadata else {}
                    if image_metadata.get('structure_guided'):
                        page_num = image_metadata.get('page_number')
                        img_index = image_metadata.get('image_index', 0)
                        key = f"image_{page_num}_{img_index}"
                        image_mapping[key] = image.id

                association_data = []

                for chunk_key, association_info in extraction_plan.get('association_pre_mapping', {}).items():
                    if chunk_key in chunk_mapping:
                        chunk_id = chunk_mapping[chunk_key]

                        for image_key in association_info['associated_images']:
                            if image_key in image_mapping:
                                image_id = image_mapping[image_key]

                                association_data.append({
                                    'complete_document_id': complete_document_id,
                                    'image_id': image_id,
                                    'document_id': chunk_id,
                                    'page_number': association_info['page_number'],
                                    'association_method': association_info['association_method'],
                                    'confidence_score': association_info['confidence_score'],
                                    'context_metadata': json.dumps({
                                        'pre_computed': True,
                                        'structure_guided': True,
                                        'postgresql_optimized': True,
                                        'created_at': datetime.now().isoformat()
                                    })
                                })

                if association_data:
                    columns = [
                        'complete_document_id', 'image_id', 'document_id',
                        'page_number', 'association_method', 'confidence_score',
                        'context_metadata'
                    ]

                    data_tuples = [
                        tuple(assoc[col] for col in columns)
                        for assoc in association_data
                    ]

                    self.bulk_insert('image_completed_document_association', data_tuples, columns)
                    associations_created = len(data_tuples)

            info_id(f"Created {associations_created} PostgreSQL-optimized associations", request_id)
            return {'associations_created': associations_created}

        except Exception as e:
            error_id(f"Error creating PostgreSQL associations: {e}", request_id)
            return {'associations_created': 0, 'error': str(e)}

    @classmethod
    def _optimize_database(cls, request_id=None):
        """Integrated database optimization that works"""
        import psycopg2
        from modules.configuration.config_env import DatabaseConfig

        try:
            config = DatabaseConfig()
            conn = psycopg2.connect(
                dbname=config.db_name,
                user=config.db_user,
                password=config.db_password,
                host=config.db_host,
                port=config.db_port
            )
            conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            cursor.execute("VACUUM ANALYZE complete_document;")
            cursor.execute("VACUUM ANALYZE documents_fts;")
            cursor.execute("VACUUM ANALYZE document;")
            cursor.close()
            conn.close()
            debug_id("PostgreSQL optimization completed successfully", request_id)
        except Exception as e:
            debug_id(f"PostgreSQL optimization failed: {e}", request_id)

    def _analyze_text_layout(self, text_dict, page_analysis, page_num, request_id=None):
        """Analyze text blocks to understand document structure."""
        try:
            for block in text_dict.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    bbox = block["bbox"]
                    text_content = ""

                    # Extract text from lines
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text_content += span.get("text", "")

                    if text_content.strip():
                        text_block = {
                            'bbox': bbox,
                            'text': text_content.strip(),
                            'font_info': self._extract_font_info(block),
                            'block_type': self._classify_text_block(text_content, block)
                        }
                        page_analysis['layout']['text_blocks'].append(text_block)

        except Exception as e:
            debug_id(f"Error analyzing text layout on page {page_num}: {e}", request_id)

    def _extract_font_info(self, block):
        """Extract font information from text block."""
        try:
            font_info = {'sizes': [], 'fonts': [], 'flags': []}

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    font_info['sizes'].append(span.get('size', 12))
                    font_info['fonts'].append(span.get('font', 'default'))
                    font_info['flags'].append(span.get('flags', 0))

            # Get dominant font characteristics
            if font_info['sizes']:
                font_info['dominant_size'] = max(set(font_info['sizes']), key=font_info['sizes'].count)
            if font_info['fonts']:
                font_info['dominant_font'] = max(set(font_info['fonts']), key=font_info['fonts'].count)

            return font_info
        except:
            return {}

    def _classify_text_block(self, text_content, block):
        """Classify text blocks by type (heading, paragraph, caption, etc.)."""
        import re

        try:
            text = text_content.strip().lower()

            # Check for headings (short, may have numbers, capitals)
            if len(text) < 100 and any(char.isupper() for char in text_content):
                if re.match(r'^[\d\.]+ ', text_content):
                    return 'numbered_heading'
                return 'heading'

            # Check for captions (often start with "Figure", "Table", etc.)
            caption_patterns = [r'^figure \d+', r'^table \d+', r'^image \d+', r'^diagram \d+']
            if any(re.match(pattern, text) for pattern in caption_patterns):
                return 'caption'

            # Check for lists
            if re.match(r'^[\u2022\u2023\u25E6\u2043•·]', text) or re.match(r'^\d+\.', text):
                return 'list_item'

            # Default to paragraph
            return 'paragraph'
        except:
            return 'paragraph'

    def _determine_layout_type(self, layout_data):
        """Determine if page is single column, two column, or mixed layout."""
        try:
            text_blocks = layout_data.get('text_blocks', [])
            if not text_blocks:
                return 'unknown'

            # Analyze horizontal positions
            left_positions = [block['bbox'][0] for block in text_blocks]
            page_width = layout_data.get('page_size', fitz.Rect()).width

            # Simple heuristic: if text starts at multiple distinct x positions
            unique_lefts = list(set([round(pos, 0) for pos in left_positions]))

            if len(unique_lefts) >= 2 and page_width > 0:
                # Check if positions suggest two columns
                if any(pos > page_width * 0.4 for pos in unique_lefts):
                    return 'two_column'

            return 'single_column'
        except:
            return 'unknown'

    def _create_page_chunk_boundaries(self, page_analysis, page_num, request_id=None):
        """Create intelligent chunk boundaries based on content structure and image positions."""
        from modules.database_manager.db_manager import ChunkBoundary

        try:
            boundaries = []
            text_blocks = page_analysis['layout']['text_blocks']
            images = page_analysis['images']

            # Sort text blocks by position (top to bottom, left to right)
            sorted_blocks = sorted(text_blocks, key=lambda b: (b['bbox'][1], b['bbox'][0]))

            current_chunk_start = None
            current_chunk_images = []

            for i, block in enumerate(sorted_blocks):
                block_bbox = block['bbox']
                block_type = block.get('block_type', 'paragraph')

                # Find images that are related to this text block
                related_images = self._find_related_images(block_bbox, images)
                current_chunk_images.extend([img.image_index for img in related_images])

                # Determine if this should be a chunk boundary
                should_split = self._should_create_chunk_boundary(block, i, sorted_blocks, related_images)

                if should_split or i == len(sorted_blocks) - 1:
                    # Create chunk boundary
                    if current_chunk_start is None:
                        current_chunk_start = block_bbox[1]  # Top of first block

                    boundary = ChunkBoundary(
                        page_number=page_num,
                        start_position=current_chunk_start,
                        end_position=block_bbox[3],  # Bottom of current block
                        chunk_type=self._determine_chunk_type(block_type),
                        associated_images=list(set(current_chunk_images)),
                        context_data={
                            'block_count': i - len(boundaries) + 1 if boundaries else i + 1,
                            'text_preview': block['text'][:100] + '...' if len(block['text']) > 100 else block['text'],
                            'layout_type': page_analysis['layout']['layout_type']
                        }
                    )
                    boundaries.append(boundary)

                    # Reset for next chunk
                    current_chunk_start = block_bbox[1] if i < len(sorted_blocks) - 1 else None
                    current_chunk_images = []

            debug_id(f"Created {len(boundaries)} chunk boundaries for page {page_num}", request_id)
            return boundaries

        except Exception as e:
            error_id(f"Error creating chunk boundaries for page {page_num}: {e}", request_id)
            return []

    def _find_related_images(self, text_bbox, images):
        """Find images that are spatially related to a text block."""
        try:
            related = []
            text_y_center = (text_bbox[1] + text_bbox[3]) / 2

            for image in images:
                img_y_center = (image.bbox[1] + image.bbox[3]) / 2

                # Consider images related if they're within a reasonable vertical distance
                vertical_distance = abs(img_y_center - text_y_center)

                # Adjust threshold based on image and text block sizes
                threshold = max(50, (text_bbox[3] - text_bbox[1]) * 2)

                if vertical_distance <= threshold:
                    related.append(image)

            return related
        except:
            return []

    def _should_create_chunk_boundary(self, block, block_index, all_blocks, related_images):
        """Determine if a chunk boundary should be created at this point."""
        try:
            block_type = block.get('block_type', 'paragraph')

            # Always split on headings
            if block_type in ['heading', 'numbered_heading']:
                return True

            # Split if there are related images (keep images with their context)
            if related_images:
                return True

            # Split on significant content changes
            if block_index > 0:
                prev_block = all_blocks[block_index - 1]
                prev_type = prev_block.get('block_type', 'paragraph')

                # Split when content type changes significantly
                if (prev_type == 'caption' and block_type == 'paragraph') or \
                        (prev_type == 'paragraph' and block_type == 'caption'):
                    return True

            # Default: don't split (continue current chunk)
            return False
        except:
            return False

    def _determine_chunk_type(self, dominant_block_type):
        """Determine the overall type of a chunk based on its content."""
        type_mapping = {
            'heading': 'section_header',
            'numbered_heading': 'section_header',
            'caption': 'image_caption',
            'paragraph': 'body_text',
            'list_item': 'list_content'
        }
        return type_mapping.get(dominant_block_type, 'body_text')

    def _create_extraction_plan(self, structure_map, request_id=None):
        """Create a comprehensive extraction plan based on the document structure analysis."""
        info_id("Creating extraction plan from structure analysis", request_id)

        try:
            plan = {
                'extraction_strategy': 'structure_guided',
                'total_chunks_planned': len(structure_map.chunk_boundaries),
                'total_images_planned': len(structure_map.image_positions),
                'page_processing_order': list(range(structure_map.total_pages)),
                'chunk_extraction_map': {},
                'image_extraction_map': {},
                'association_pre_mapping': {},
                'processing_hints': {}
            }

            # Create chunk extraction mapping
            for i, chunk_boundary in enumerate(structure_map.chunk_boundaries):
                chunk_id = f"chunk_{chunk_boundary.page_number}_{i}"
                plan['chunk_extraction_map'][chunk_id] = {
                    'page_number': chunk_boundary.page_number,
                    'extraction_bbox': (0, chunk_boundary.start_position, 9999, chunk_boundary.end_position),
                    'chunk_type': chunk_boundary.chunk_type,
                    'expected_images': chunk_boundary.associated_images,
                    'context_data': chunk_boundary.context_data
                }

            # Create image extraction mapping
            for i, image_pos in enumerate(structure_map.image_positions):
                image_id = f"image_{image_pos.page_number}_{image_pos.image_index}"
                plan['image_extraction_map'][image_id] = {
                    'page_number': image_pos.page_number,
                    'image_index': image_pos.image_index,
                    'extraction_bbox': image_pos.bbox,
                    'estimated_size': image_pos.estimated_size,
                    'content_type': image_pos.content_type
                }

            # Create pre-mapping for associations
            for chunk_id, chunk_data in plan['chunk_extraction_map'].items():
                associated_image_ids = []
                for img_index in chunk_data['expected_images']:
                    # Find corresponding image_id
                    for image_id, image_data in plan['image_extraction_map'].items():
                        if (image_data['page_number'] == chunk_data['page_number'] and
                                image_data['image_index'] == img_index):
                            associated_image_ids.append(image_id)

                if associated_image_ids:
                    plan['association_pre_mapping'][chunk_id] = {
                        'associated_images': associated_image_ids,
                        'confidence_score': 0.9,  # High confidence from structure analysis
                        'association_method': 'structure_guided',
                        'page_number': chunk_data['page_number']
                    }

            info_id(f"Extraction plan created: {plan['total_chunks_planned']} chunks, "
                    f"{plan['total_images_planned']} images", request_id)

            return plan

        except Exception as e:
            error_id(f"Error creating extraction plan: {e}", request_id)
            return {}

    def _perform_postgresql_guided_extraction(self, file_path: str, structure_map, metadata: Dict[str, Any],
                                              request_id=None):
        """Perform text and image extraction guided by the structure map."""
        try:
            from modules.emtacdb.emtacdb_fts import CompleteDocument
            import json

            # Create complete document record
            complete_doc = CompleteDocument(
                title=metadata.get('title', 'Unknown Document'),
                file_path=file_path,
                content="",  # Will be filled by chunks
                rev="R0"
            )

            with self.transaction():
                self.session.add(complete_doc)
                self.session.flush()
                complete_document_id = complete_doc.id

                # Extract chunks using the structure map guidance
                chunks_created = self._extract_chunks_with_guidance(
                    file_path, complete_document_id, structure_map, self.session, request_id
                )

                # Extract images using the structure map guidance
                images_extracted = self._extract_images_with_guidance(
                    file_path, complete_document_id, structure_map, self.session, request_id
                )

                return {
                    'success': True,
                    'complete_document_id': complete_document_id,
                    'chunks_created': chunks_created,
                    'images_extracted': images_extracted
                }

        except Exception as e:
            error_id(f"Error in guided extraction: {e}", request_id)
            return {'success': False, 'error': str(e)}

    def _extract_chunks_with_guidance(self, file_path: str, complete_document_id: int,
                                      structure_map, session, request_id=None) -> int:
        """Extract text chunks using structure guidance."""
        from modules.emtacdb.emtacdb_fts import Document
        import json

        try:
            doc = fitz.open(file_path)
            chunks_created = 0

            for chunk_id, chunk_data in structure_map.extraction_plan['chunk_extraction_map'].items():
                try:
                    page_num = chunk_data['page_number']
                    page = doc[page_num]

                    # Extract text from the specified area
                    extraction_rect = fitz.Rect(chunk_data['extraction_bbox'])
                    chunk_text = page.get_text("text", clip=extraction_rect)

                    if chunk_text.strip():
                        # Create Document record with structure guidance metadata
                        document_chunk = Document(
                            name=f"{chunk_id}_{chunk_data['chunk_type']}",
                            file_path=file_path,
                            content=chunk_text.strip(),
                            complete_document_id=complete_document_id,
                            rev="R0",
                            metadata=json.dumps({
                                'page_number': page_num,
                                'chunk_type': chunk_data['chunk_type'],
                                'structure_guided': True,
                                'expected_images': chunk_data['expected_images'],
                                'extraction_bbox': chunk_data['extraction_bbox'],
                                'context_data': chunk_data['context_data']
                            })
                        )

                        session.add(document_chunk)
                        chunks_created += 1

                except Exception as e:
                    error_id(f"Error extracting chunk {chunk_id}: {e}", request_id)
                    continue

            doc.close()
            debug_id(f"Created {chunks_created} chunks with structure guidance", request_id)
            return chunks_created

        except Exception as e:
            error_id(f"Error in guided chunk extraction: {e}", request_id)
            return 0

class DocumentStructureManager(PostgreSQLDocumentStructureManager):
    """
    Backward compatible document structure manager.
    Uses PostgreSQL backend with enhanced structure analysis.
    """

    def __init__(self, session=None, request_id=None):
        super().__init__(session, request_id)
        info_id("Using PostgreSQL-backed document structure manager", self.request_id)

    def analyze_document(self, file_path: str) -> DocumentStructureMap:
        """Backward compatible method name."""
        return self.analyze_document_structure(file_path, self.request_id)

    def guided_extraction(self, file_path: str, metadata: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], int]:
        """Backward compatible method name."""
        return self.guided_extraction_with_postgresql(file_path, metadata, self.request_id)

def create_document_structure_tables(db_config: DatabaseConfig):
    """
    Create necessary tables for document structure analysis.
    Call this during your database setup.
    """
    try:
        with db_config.main_session() as session:
            # Create structure analysis storage table
            session.execute(text("""
                CREATE TABLE IF NOT EXISTS document_structure_analysis (
                    id SERIAL PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    analysis_date TIMESTAMP DEFAULT NOW(),
                    total_pages INTEGER,
                    total_images INTEGER,
                    total_chunks INTEGER,
                    structure_data JSONB,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(file_path, analysis_date)
                )
            """))

            # Create indexes for performance
            session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_doc_structure_file_path 
                ON document_structure_analysis(file_path)
            """))

            session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_doc_structure_analysis_date 
                ON document_structure_analysis(analysis_date DESC)
            """))

            # Create GIN index for JSONB structure_data
            session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_doc_structure_data_gin 
                ON document_structure_analysis USING gin(structure_data)
            """))

            session.commit()
            print("Document structure analysis tables created successfully")

    except Exception as e:
        print(f"Error creating document structure tables: {e}")
        raise

def get_structure_analysis_stats(db_config: DatabaseConfig) -> Dict[str, Any]:
    """
    Get statistics about stored structure analyses.
    """
    try:
        with db_config.main_session() as session:
            result = session.execute(text("""
                SELECT 
                    COUNT(*) as total_analyses,
                    AVG(total_pages) as avg_pages,
                    AVG(total_images) as avg_images,
                    AVG(total_chunks) as avg_chunks,
                    MAX(analysis_date) as latest_analysis,
                    MIN(analysis_date) as earliest_analysis
                FROM document_structure_analysis
            """)).fetchone()

            if result:
                return {
                    'total_analyses': result[0],
                    'average_pages': float(result[1]) if result[1] else 0,
                    'average_images': float(result[2]) if result[2] else 0,
                    'average_chunks': float(result[3]) if result[3] else 0,
                    'latest_analysis': result[4].isoformat() if result[4] else None,
                    'earliest_analysis': result[5].isoformat() if result[5] else None
                }

            return {}

    except Exception as e:
        print(f"Error getting structure analysis stats: {e}")
        return {}

# ==========================================
# EXAMPLE USAGE WITH YOUR EXISTING CODE
# ==========================================

def example_integration_with_existing_code():
    """
    Example of how to integrate structure analysis with your existing db_manager.py patterns.
    """

    # Using your existing database patterns
    with PostgreSQLDatabaseManager() as db_manager:
        # Create structure manager using the same session
        structure_manager = PostgreSQLDocumentStructureManager(
            session=db_manager.session,
            request_id=db_manager.request_id
        )

        # Analyze document structure
        file_path = "/path/to/document.pdf"
        structure_map = structure_manager.analyze_document_structure(file_path)

        # Use the structure map for guided extraction
        metadata = {"title": "Technical Manual", "department": "Engineering"}
        success, result, status = structure_manager.guided_extraction_with_postgresql(
            file_path, metadata
        )

        if success:
            print(f"Structure-guided processing completed:")
            print(f"- Document ID: {result['complete_document_id']}")
            print(f"- Chunks created: {result['chunks_created']}")
            print(f"- Images extracted: {result['images_extracted']}")
            print(f"- Associations created: {result['associations_created']}")


def enhanced_database_setup():
    """
    Enhanced database setup including structure analysis tables.
    Add this to your existing database initialization.
    """
    db_config = DatabaseConfig()

    # Your existing database setup...

    # Add structure analysis tables
    create_document_structure_tables(db_config)

    # Get current stats
    stats = get_structure_analysis_stats(db_config)
    print(f"Structure analysis database ready. Current stats: {stats}")

class PostgreSQLRelationshipManager(PostgreSQLDatabaseManager):
    """Enhanced PostgreSQL relationship manager with concurrent processing and modern patterns."""

    def associate_parts_with_images_by_title(self, part_ids=None, position_id=None, use_concurrent=True,
                                             fuzzy_matching=True):
        """
        Enhanced part-image association with concurrent processing and fuzzy matching.

        Args:
            part_ids: List of part IDs to process (None for all parts)
            position_id: Optional position ID to include in associations
            use_concurrent: Use concurrent processing for large datasets
            fuzzy_matching: Use PostgreSQL fuzzy matching for better results

        Returns:
            Dictionary mapping part IDs to lists of created associations
        """
        info_id("Starting enhanced PostgreSQL part-image association process", self.request_id)
        result = {}

        try:
            with self.transaction():
                # Get parts to process
                if part_ids is None:
                    parts = self.session.query(Part).all()
                else:
                    parts = self.session.query(Part).filter(Part.id.in_(part_ids)).all()

                info_id(f"Processing {len(parts)} parts for image associations", self.request_id)

                # Use concurrent processing for large datasets
                if use_concurrent and len(parts) > 10:
                    result = self._associate_parts_concurrent(parts, position_id, fuzzy_matching)
                else:
                    # Sequential processing
                    for part in parts:
                        associations = self._associate_single_part(part, position_id, fuzzy_matching)
                        result[part.id] = associations

                # Optimize database after bulk operations
                self._optimize_associations()

            return result
        except Exception as e:
            error_id(f"Error in enhanced PostgreSQL part-image association: {str(e)}", self.request_id, exc_info=True)
            raise

    def _associate_parts_concurrent(self, parts, position_id, fuzzy_matching):
        """Concurrent processing for large part datasets."""
        result = {}
        max_workers = min(len(parts), 4)

        info_id(f"Using {max_workers} concurrent workers for PostgreSQL part association", self.request_id)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._associate_part_worker, part, position_id, fuzzy_matching): part
                for part in parts
            }

            for future in as_completed(futures):
                part = futures[future]
                try:
                    associations = future.result()
                    result[part.id] = associations
                    debug_id(f"Completed associations for part {part.part_number}", self.request_id)
                except Exception as e:
                    error_id(f"Error associating part {part.id}: {e}", self.request_id)
                    result[part.id] = []

        return result

    def _associate_part_worker(self, part, position_id, fuzzy_matching):
        """Worker method for concurrent part association."""
        # Create a new manager instance for this worker
        with PostgreSQLDatabaseManager(request_id=self.request_id) as worker_manager:
            # Recreate the part object in the new session
            worker_part = worker_manager.session.query(Part).filter(Part.id == part.id).first()
            if worker_part:
                return self._associate_single_part_enhanced(worker_manager.session, worker_part, position_id,
                                                            fuzzy_matching)
            return []

    def _associate_single_part(self, part, position_id=None, fuzzy_matching=True):
        """Enhanced helper method to associate a single part with matching images."""
        return self._associate_single_part_enhanced(self.session, part, position_id, fuzzy_matching)

    def _associate_single_part_enhanced(self, session, part, position_id=None, fuzzy_matching=True):
        """Enhanced single part association with fuzzy matching."""
        created = []

        try:
            # Build query based on fuzzy matching preference
            if fuzzy_matching:
                try:
                    # Use PostgreSQL similarity for fuzzy matching
                    matching_images = session.query(Image).filter(
                        text("similarity(title, :part_number) > 0.3")
                    ).params(part_number=part.part_number).all()
                except Exception:
                    # Fallback to case-insensitive like matching
                    matching_images = session.query(Image).filter(
                        Image.title.ilike(f"%{part.part_number}%")
                    ).all()
            else:
                # Exact case-insensitive matching
                matching_images = session.query(Image).filter(
                    Image.title.ilike(part.part_number)
                ).all()

            info_id(f"Found {len(matching_images)} images matching part {part.part_number}", self.request_id)

            if not matching_images:
                return created

            # Batch check for existing associations
            existing_associations = set()
            query = session.query(PartsPositionImageAssociation).filter(
                and_(
                    PartsPositionImageAssociation.part_id == part.id,
                    PartsPositionImageAssociation.image_id.in_([img.id for img in matching_images])
                )
            )

            if position_id is not None:
                query = query.filter(PartsPositionImageAssociation.position_id == position_id)

            for assoc in query.all():
                existing_associations.add((assoc.image_id, assoc.position_id))

            # Create new associations
            for image in matching_images:
                key = (image.id, position_id)
                if key not in existing_associations:
                    assoc = PartsPositionImageAssociation(
                        part_id=part.id,
                        image_id=image.id,
                        position_id=position_id
                    )
                    session.add(assoc)
                    created.append(assoc)

            if created:
                session.flush()
                debug_id(f"Created {len(created)} new associations for part {part.part_number}", self.request_id)

        except Exception as e:
            error_id(f"Error in enhanced single part association: {e}", self.request_id)

        return created

    def associate_drawings_with_parts_by_number(self, batch_size=100):
        """
        Enhanced drawing-part association with batch processing and better performance.
        Handles multiple comma-separated part numbers per drawing.

        Args:
            batch_size: Number of drawings to process in each batch

        Returns:
            Dict mapping drawing_id to list of created associations
        """
        info_id("Starting enhanced PostgreSQL drawing-part association process", self.request_id)

        try:
            with self.transaction():
                # Get all drawings with spare part numbers using optimized query
                drawings_query = self.session.query(Drawing).filter(
                    and_(
                        Drawing.drw_spare_part_number.isnot(None),
                        Drawing.drw_spare_part_number != ''
                    )
                )

                total_count = drawings_query.count()
                info_id(f"Found {total_count} drawings with spare part numbers", self.request_id)

                associations_by_drawing = {}
                processed = 0

                # Process in batches for better memory management
                for offset in range(0, total_count, batch_size):
                    batch_drawings = drawings_query.offset(offset).limit(batch_size).all()

                    for drawing in batch_drawings:
                        try:
                            drawing_associations = self._process_single_drawing_enhanced(drawing)
                            if drawing_associations:
                                associations_by_drawing[drawing.id] = drawing_associations

                            processed += 1
                            if processed % 50 == 0:
                                info_id(f"Processed {processed}/{total_count} drawings", self.request_id)

                        except Exception as e:
                            error_id(f"Error processing drawing {drawing.id}: {e}", self.request_id)
                            continue

                # Optimize database after bulk operations
                self._optimize_associations()

                info_id(f"Created new associations for {len(associations_by_drawing)} drawings", self.request_id)
                return associations_by_drawing

        except Exception as e:
            error_id(f"Error in enhanced PostgreSQL drawing-part association: {str(e)}", self.request_id, exc_info=True)
            raise

    def _process_single_drawing_enhanced(self, drawing):
        """Enhanced processing of a single drawing for part associations."""
        if not drawing.drw_spare_part_number or not drawing.drw_spare_part_number.strip():
            return []

        debug_id(f"Processing drawing {drawing.drw_number} with spare part number(s): {drawing.drw_spare_part_number}",
                 self.request_id)

        # Split and clean part numbers
        part_numbers = [pn.strip() for pn in drawing.drw_spare_part_number.split(',') if pn.strip()]
        drawing_associations = []

        if not part_numbers:
            return drawing_associations

        # Get all matching parts in optimized queries
        all_matching_parts = []
        for part_number in part_numbers:
            try:
                # Use case-insensitive matching with wildcards
                matching_parts = self.session.query(Part).filter(
                    Part.part_number.ilike(f"%{part_number}%")
                ).all()
                all_matching_parts.extend(matching_parts)
                debug_id(f"Found {len(matching_parts)} parts matching '{part_number}'", self.request_id)
            except Exception as e:
                debug_id(f"Error searching for part number '{part_number}': {e}", self.request_id)
                continue

        if not all_matching_parts:
            return drawing_associations

        # Get existing associations to avoid duplicates
        existing_part_ids = set()
        if all_matching_parts:
            existing_assocs = self.session.query(DrawingPartAssociation).filter(
                and_(
                    DrawingPartAssociation.drawing_id == drawing.id,
                    DrawingPartAssociation.part_id.in_([p.id for p in all_matching_parts])
                )
            ).all()

            existing_part_ids = {assoc.part_id for assoc in existing_assocs}

        # Create new associations
        for part in all_matching_parts:
            if part.id not in existing_part_ids:
                association = DrawingPartAssociation(
                    drawing_id=drawing.id,
                    part_id=part.id
                )
                self.session.add(association)
                drawing_associations.append(association)
                debug_id(f"Created association between drawing {drawing.drw_number} and part {part.part_number}",
                         self.request_id)

        if drawing_associations:
            self.session.flush()

        return drawing_associations

    def bulk_associate_parts_images(self, associations_data, batch_size=1000):
        """
        Enhanced bulk create part-image associations with better performance.

        Args:
            associations_data: List of dicts with keys: part_id, image_id, position_id
            batch_size: Number of associations to process in each batch
        """
        try:
            if not associations_data:
                warning_id("No association data provided for bulk operation", self.request_id)
                return

            info_id(f"Processing {len(associations_data)} associations in batches of {batch_size}", self.request_id)

            with self.transaction():
                # Process in batches to avoid memory issues
                for i in range(0, len(associations_data), batch_size):
                    batch = associations_data[i:i + batch_size]
                    self._process_association_batch(batch)

                    info_id(
                        f"Processed batch {i // batch_size + 1}/{(len(associations_data) + batch_size - 1) // batch_size}",
                        self.request_id)

                # Optimize after bulk operations
                self._optimize_associations()

        except Exception as e:
            error_id(f"Error in enhanced bulk part-image association: {str(e)}", self.request_id, exc_info=True)
            raise

    def _process_association_batch(self, batch):
        """Process a batch of associations with optimized duplicate checking."""
        if not batch:
            return

        # Build efficient query to check existing associations
        conditions = []
        for assoc_data in batch:
            condition = and_(
                PartsPositionImageAssociation.part_id == assoc_data['part_id'],
                PartsPositionImageAssociation.image_id == assoc_data['image_id'],
                PartsPositionImageAssociation.position_id == assoc_data.get('position_id')
            )
            conditions.append(condition)

        # Get existing associations in one query
        existing_keys = set()
        if conditions:
            existing_assocs = self.session.query(PartsPositionImageAssociation).filter(
                or_(*conditions)
            ).all()

            for assoc in existing_assocs:
                existing_keys.add((assoc.part_id, assoc.image_id, assoc.position_id))

        # Create only new associations
        new_associations = []
        for assoc_data in batch:
            key = (assoc_data['part_id'], assoc_data['image_id'], assoc_data.get('position_id'))
            if key not in existing_keys:
                new_associations.append(PartsPositionImageAssociation(**assoc_data))

        if new_associations:
            self.session.add_all(new_associations)
            self.session.flush()
            debug_id(f"Created {len(new_associations)} new associations in batch", self.request_id)

    def _optimize_associations(self):
        """Optimize association tables after bulk operations."""
        try:
            self._analyze_table('parts_position_image_association')
            self._analyze_table('drawing_part_association')
        except Exception as e:
            debug_id(f"Association optimization skipped: {e}", self.request_id)


class PostgreSQLDuplicateManager(PostgreSQLDatabaseManager):
    """Enhanced duplicate detection and management with modern patterns and fuzzy matching."""

    def find_duplicate_parts(self, threshold=0.9, use_fuzzy_matching=True, batch_size=500):
        """
        Enhanced duplicate detection with configurable strategies and batch processing.
        Uses PostgreSQL's text search capabilities.

        Args:
            threshold: Similarity threshold (0.0-1.0)
            use_fuzzy_matching: Whether to use PostgreSQL's fuzzy string matching
            batch_size: Limit results to manage memory

        Returns:
            List of dictionaries containing potential duplicate part information
        """
        info_id(
            f"Finding duplicate parts with threshold {threshold} using {'fuzzy' if use_fuzzy_matching else 'exact'} matching",
            self.request_id)

        try:
            if use_fuzzy_matching:
                return self._find_duplicates_fuzzy(threshold, batch_size)
            else:
                return self._find_duplicates_exact(batch_size)

        except Exception as e:
            error_id(f"Error finding duplicate parts: {str(e)}", self.request_id, exc_info=True)
            raise

    def _find_duplicates_fuzzy(self, threshold, batch_size):
        """Use PostgreSQL's similarity function for fuzzy duplicate detection."""
        try:
            # Enable pg_trgm extension if not already enabled
            with self.savepoint():
                self.session.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))

            sql = text("""
                SELECT p1.id as id1, p1.part_number as part1, 
                       p2.id as id2, p2.part_number as part2,
                       similarity(p1.part_number, p2.part_number) as sim_score
                FROM part p1
                JOIN part p2 ON p1.id < p2.id
                WHERE similarity(p1.part_number, p2.part_number) > :threshold
                ORDER BY sim_score DESC
                LIMIT :batch_size
            """)

            result = self.execute_raw_sql(sql, {'threshold': threshold, 'batch_size': batch_size})
            duplicates = [
                {
                    'id1': row[0], 'part1': row[1],
                    'id2': row[2], 'part2': row[3],
                    'similarity': float(row[4]),
                    'match_type': 'fuzzy'
                }
                for row in result.fetchall()
            ]

            info_id(f"Found {len(duplicates)} potential fuzzy duplicate pairs", self.request_id)
            return duplicates

        except Exception as e:
            error_id(f"PostgreSQL fuzzy matching failed, falling back to exact matching: {e}", self.request_id)
            return self._find_duplicates_exact(batch_size)

    def _find_duplicates_exact(self, batch_size):
        """Exact matching approach with enhanced text processing."""
        try:
            sql = text("""
                SELECT p1.id as id1, p1.part_number as part1,
                       p2.id as id2, p2.part_number as part2,
                       1.0 as sim_score
                FROM part p1
                JOIN part p2 ON p1.id < p2.id
                WHERE LOWER(TRIM(p1.part_number)) = LOWER(TRIM(p2.part_number))
                   OR REPLACE(LOWER(TRIM(p1.part_number)), '-', '') = REPLACE(LOWER(TRIM(p2.part_number)), '-', '')
                   OR REPLACE(REPLACE(LOWER(TRIM(p1.part_number)), '-', ''), ' ', '') = 
                      REPLACE(REPLACE(LOWER(TRIM(p2.part_number)), '-', ''), ' ', '')
                ORDER BY p1.part_number
                LIMIT :batch_size
            """)

            result = self.execute_raw_sql(sql, {'batch_size': batch_size})
            duplicates = [
                {
                    'id1': row[0], 'part1': row[1],
                    'id2': row[2], 'part2': row[3],
                    'similarity': float(row[4]),
                    'match_type': 'exact'
                }
                for row in result.fetchall()
            ]

            info_id(f"Found {len(duplicates)} exact duplicate pairs", self.request_id)
            return duplicates

        except Exception as e:
            error_id(f"Exact duplicate detection failed: {e}", self.request_id)
            return []

    def merge_duplicate_parts(self, source_id, target_id, fields_to_merge=None, dry_run=False):
        """
        Enhanced part merging with transaction safety and dry-run capability.
        Merges two duplicate parts using PostgreSQL transactions.

        Args:
            source_id: ID of the source part (will be merged into target)
            target_id: ID of the target part (will be kept)
            fields_to_merge: List of fields to merge (None for all non-null fields)
            dry_run: If True, only report what would be done without making changes

        Returns:
            Dictionary with merge results and statistics
        """
        info_id(f"{'[DRY RUN] ' if dry_run else ''}Merging part {source_id} into part {target_id}", self.request_id)

        try:
            if dry_run:
                return self._merge_parts_dry_run(source_id, target_id, fields_to_merge)
            else:
                return self._merge_parts_execute(source_id, target_id, fields_to_merge)

        except Exception as e:
            error_id(f"Error merging parts: {str(e)}", self.request_id, exc_info=True)
            raise

    def _merge_parts_dry_run(self, source_id, target_id, fields_to_merge):
        """Dry run mode - analyze what would be merged without making changes."""
        source_part = self.session.query(Part).filter(Part.id == source_id).first()
        target_part = self.session.query(Part).filter(Part.id == target_id).first()

        if not source_part:
            raise ValueError(f"Source part {source_id} not found")
        if not target_part:
            raise ValueError(f"Target part {target_id} not found")

        # Count associations
        associations_count = self._count_part_associations(source_id)

        # Analyze field merges
        merged_fields = self._analyze_field_merges(source_part, target_part, fields_to_merge)

        merge_stats = {
            'source_part': source_part.part_number,
            'target_part': target_part.part_number,
            'associations_to_update': associations_count,
            'fields_to_merge': merged_fields,
            'dry_run': True,
            'would_succeed': True
        }

        info_id(f"[DRY RUN] Merge analysis completed: {merge_stats}", self.request_id)
        return merge_stats

    def _merge_parts_execute(self, source_id, target_id, fields_to_merge):
        """Execute the actual merge operation."""
        with self.transaction():
            source_part = self.session.query(Part).filter(Part.id == source_id).first()
            target_part = self.session.query(Part).filter(Part.id == target_id).first()

            if not source_part:
                raise ValueError(f"Source part {source_id} not found")
            if not target_part:
                raise ValueError(f"Target part {target_id} not found")

            merge_stats = {
                'source_part': source_part.part_number,
                'target_part': target_part.part_number,
                'associations_updated': 0,
                'fields_merged': [],
                'dry_run': False
            }

            # Update related associations to point to target part
            merge_stats['associations_updated'] = self._update_part_associations(source_id, target_id)

            # Merge specified fields or all non-null fields
            merged_fields = self._merge_part_fields(source_part, target_part, fields_to_merge)
            merge_stats['fields_merged'] = merged_fields

            # Delete the source part
            self.session.delete(source_part)
            self.session.flush()

            info_id(f"Successfully merged part {source_id} into {target_id}: {merge_stats}", self.request_id)
            return merge_stats

    def _update_part_associations(self, source_id, target_id):
        """Enhanced association updates with better conflict handling."""
        total_updated = 0

        try:
            # Update PartsPositionImageAssociation with conflict resolution
            result1 = self.execute_raw_sql("""
                UPDATE parts_position_image_association 
                SET part_id = :target_id 
                WHERE part_id = :source_id
                AND NOT EXISTS (
                    SELECT 1 FROM parts_position_image_association ppia2
                    WHERE ppia2.part_id = :target_id 
                    AND ppia2.image_id = parts_position_image_association.image_id
                    AND COALESCE(ppia2.position_id, -1) = COALESCE(parts_position_image_association.position_id, -1)
                )
            """, {'source_id': source_id, 'target_id': target_id})

            updated1 = result1.rowcount if hasattr(result1, 'rowcount') else 0

            # Update DrawingPartAssociation with conflict resolution
            result2 = self.execute_raw_sql("""
                UPDATE drawing_part_association 
                SET part_id = :target_id 
                WHERE part_id = :source_id
                AND NOT EXISTS (
                    SELECT 1 FROM drawing_part_association dpa2
                    WHERE dpa2.part_id = :target_id 
                    AND dpa2.drawing_id = drawing_part_association.drawing_id
                )
            """, {'source_id': source_id, 'target_id': target_id})

            updated2 = result2.rowcount if hasattr(result2, 'rowcount') else 0

            # Delete remaining duplicate associations
            self.execute_raw_sql("""
                DELETE FROM parts_position_image_association 
                WHERE part_id = :source_id
            """, {'source_id': source_id})

            self.execute_raw_sql("""
                DELETE FROM drawing_part_association 
                WHERE part_id = :source_id
            """, {'source_id': source_id})

            total_updated = updated1 + updated2
            debug_id(f"Updated {total_updated} associations", self.request_id)

        except Exception as e:
            error_id(f"Error updating part associations: {str(e)}", self.request_id, exc_info=True)
            raise

        return total_updated

    def _count_part_associations(self, part_id):
        """Count associations for dry-run analysis."""
        try:
            count1 = self.session.query(PartsPositionImageAssociation).filter(
                PartsPositionImageAssociation.part_id == part_id
            ).count()

            count2 = self.session.query(DrawingPartAssociation).filter(
                DrawingPartAssociation.part_id == part_id
            ).count()

            return count1 + count2
        except Exception as e:
            warning_id(f"Could not count associations: {e}", self.request_id)
            return 0

    def _analyze_field_merges(self, source_part, target_part, fields_to_merge):
        """Analyze which fields would be merged in dry-run mode."""
        field_analysis = []

        try:
            part_columns = [column.name for column in source_part.__table__.columns]

            if fields_to_merge:
                fields_to_process = [f for f in fields_to_merge if f in part_columns and f != 'id']
            else:
                fields_to_process = [f for f in part_columns if f != 'id']

            for field_name in fields_to_process:
                source_value = getattr(source_part, field_name, None)
                target_value = getattr(target_part, field_name, None)

                if source_value and not target_value:
                    field_analysis.append({
                        'field': field_name,
                        'current_target_value': target_value,
                        'new_value_from_source': source_value,
                        'action': 'merge'
                    })
                elif source_value and target_value and source_value != target_value:
                    field_analysis.append({
                        'field': field_name,
                        'current_target_value': target_value,
                        'source_value': source_value,
                        'action': 'conflict - keeping target'
                    })

        except Exception as e:
            warning_id(f"Error analyzing field merges: {e}", self.request_id)

        return field_analysis

    def _merge_part_fields(self, source_part, target_part, fields_to_merge):
        """Execute field merging with conflict resolution."""
        merged_fields = []

        try:
            part_columns = [column.name for column in source_part.__table__.columns]

            if fields_to_merge:
                fields_to_process = [f for f in fields_to_merge if f in part_columns and f != 'id']
            else:
                fields_to_process = [f for f in part_columns if f != 'id']

            for field_name in fields_to_process:
                source_value = getattr(source_part, field_name, None)
                target_value = getattr(target_part, field_name, None)

                # Merge non-null source values into null target fields
                if source_value and not target_value:
                    setattr(target_part, field_name, source_value)
                    merged_fields.append({
                        'field': field_name,
                        'old_value': target_value,
                        'new_value': source_value
                    })

            if merged_fields:
                self.session.flush()

        except Exception as e:
            warning_id(f"Error merging fields: {e}", self.request_id)

        return merged_fields


class EnhancedExcelToPostgreSQLMapper:
    """Enhanced Excel to PostgreSQL mapper with modern patterns and better error handling."""

    def __init__(self, excel_path, db_config=None):
        """
        Initialize the mapper with an Excel file path and PostgreSQL database configuration.

        Args:
            excel_path: Path to the Excel file
            db_config: DatabaseConfig instance. If None, a new one will be created.
        """
        self.request_id = set_request_id()
        info_id("Initializing EnhancedExcelToPostgreSQLMapper", self.request_id)

        self.excel_path = excel_path
        self.db_config = db_config if db_config else DatabaseConfig()

        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Excel file not found: {excel_path}")

        debug_id(f"Mapper initialized with Excel file: {excel_path}", self.request_id)

    def infer_postgresql_type(self, pandas_dtype):
        """Enhanced PostgreSQL type inference from pandas dtype."""
        if pd.api.types.is_integer_dtype(pandas_dtype):
            return 'INTEGER'
        elif pd.api.types.is_float_dtype(pandas_dtype):
            return 'NUMERIC'
        elif pd.api.types.is_bool_dtype(pandas_dtype):
            return 'BOOLEAN'
        elif pd.api.types.is_datetime64_any_dtype(pandas_dtype):
            return 'TIMESTAMP'
        else:
            return 'TEXT'

    @with_request_id
    def prompt_for_mapping(self, df):
        """
        Enhanced prompt for column mapping with better validation.
        Decorated with request ID tracking.
        """
        mapping = {}
        type_overrides = {}

        info_id("Excel columns found:", self.request_id)
        for i, col in enumerate(df.columns):
            dtype = self.infer_postgresql_type(df[col].dtype)
            sample_data = df[col].dropna().head(3).tolist() if not df[col].empty else []
            debug_id(f"  {i + 1}. '{col}' (type: {dtype}, samples: {sample_data})", self.request_id)

        info_id("For each Excel column, specify the PostgreSQL column name to map to (leave blank to skip):",
                self.request_id)

        for col in df.columns:
            dtype = self.infer_postgresql_type(df[col].dtype)
            mapped_col = input(f"Map Excel column '{col}' to PostgreSQL column (or blank to skip): ").strip()
            if mapped_col:
                type_choice = input(
                    f" - Data type for '{mapped_col}'? [INTEGER/NUMERIC/TEXT/BOOLEAN/TIMESTAMP, default: {dtype}]: "
                ).strip().upper()
                valid_types = ['INTEGER', 'NUMERIC', 'TEXT', 'BOOLEAN', 'TIMESTAMP']
                type_overrides[mapped_col] = type_choice if type_choice in valid_types else dtype
                mapping[col] = mapped_col
                debug_id(f"Mapped '{col}' to '{mapped_col}' with type {type_overrides[mapped_col]}", self.request_id)

        return mapping, type_overrides

    def create_mapping_table(self, session):
        """Enhanced mapping table creation with better error handling."""
        with log_timed_operation("create_mapping_table", self.request_id):
            try:
                session.execute(text("""
                CREATE TABLE IF NOT EXISTS excel_postgresql_mapping (
                    id SERIAL PRIMARY KEY,
                    mapping_name TEXT UNIQUE,
                    excel_file TEXT NOT NULL,
                    excel_sheet TEXT NOT NULL,
                    postgresql_table TEXT NOT NULL,
                    column_mapping JSONB NOT NULL,
                    column_types JSONB NOT NULL,
                    row_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """))

                # Create indexes for better performance
                session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_excel_postgresql_mapping_name 
                ON excel_postgresql_mapping(mapping_name)
                """))

                session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_excel_postgresql_mapping_table 
                ON excel_postgresql_mapping(postgresql_table)
                """))

                session.commit()
                debug_id("Enhanced PostgreSQL mapping table created or already exists", self.request_id)
            except Exception as e:
                session.rollback()
                error_id(f"Error creating PostgreSQL mapping table: {str(e)}", self.request_id)
                raise

    def store_mapping(self, session, mapping_name, excel_file, excel_sheet, postgresql_table, mapping, type_overrides,
                      row_count=0):
        """Enhanced mapping storage with upsert capability."""
        with log_timed_operation("store_mapping", self.request_id):
            try:
                # Use upsert to handle duplicate mapping names
                sql = text("""
                INSERT INTO excel_postgresql_mapping 
                (mapping_name, excel_file, excel_sheet, postgresql_table, column_mapping, column_types, row_count, created_at, updated_at)
                VALUES (:name, :file, :sheet, :table, :mapping, :types, :row_count, :created_at, :updated_at)
                ON CONFLICT (mapping_name) DO UPDATE SET
                    excel_file = EXCLUDED.excel_file,
                    excel_sheet = EXCLUDED.excel_sheet,
                    postgresql_table = EXCLUDED.postgresql_table,
                    column_mapping = EXCLUDED.column_mapping,
                    column_types = EXCLUDED.column_types,
                    row_count = EXCLUDED.row_count,
                    updated_at = EXCLUDED.updated_at
                """)

                session.execute(sql, {
                    'name': mapping_name,
                    'file': excel_file,
                    'sheet': excel_sheet,
                    'table': postgresql_table,
                    'mapping': json.dumps(mapping),
                    'types': json.dumps(type_overrides),
                    'row_count': row_count,
                    'created_at': datetime.now(),
                    'updated_at': datetime.now()
                })
                session.commit()
                info_id(f"PostgreSQL mapping information stored/updated for '{mapping_name}'", self.request_id)
            except Exception as e:
                session.rollback()
                error_id(f"Error storing PostgreSQL mapping information: {str(e)}", self.request_id)
                raise

    def create_table(self, session, table_name, mapping, type_overrides):
        """Enhanced table creation with better column handling."""
        with log_timed_operation(f"create_table_{table_name}", self.request_id):
            try:
                columns = []
                for excel_col, postgresql_col in mapping.items():
                    col_type = type_overrides[postgresql_col]
                    columns.append(f'"{postgresql_col}" {col_type}')

                col_defs = ", ".join(columns)
                sql = f'''
                CREATE TABLE IF NOT EXISTS "{table_name}" (
                    id SERIAL PRIMARY KEY, 
                    {col_defs},
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                '''

                session.execute(text(sql))

                # Create basic indexes
                for postgresql_col in mapping.values():
                    try:
                        index_sql = f'CREATE INDEX IF NOT EXISTS idx_{table_name}_{postgresql_col} ON "{table_name}" ("{postgresql_col}")'
                        session.execute(text(index_sql))
                    except:
                        pass  # Skip if index creation fails

                session.commit()
                info_id(f"Created enhanced PostgreSQL table '{table_name}' with {len(columns)} columns",
                        self.request_id)
            except Exception as e:
                session.rollback()
                error_id(f"Error creating PostgreSQL table '{table_name}': {str(e)}", self.request_id)
                raise

    def insert_data(self, session, table_name, df, mapping, batch_size=1000):
        """Enhanced data insertion with batch processing and better error handling."""
        with log_timed_operation(f"insert_data_{table_name}", self.request_id):
            try:
                mapped_cols = list(mapping.keys())
                postgresql_cols = [mapping[col] for col in mapped_cols]

                total_rows = len(df)
                info_id(f"Inserting {total_rows} rows into PostgreSQL table '{table_name}' in batches of {batch_size}",
                        self.request_id)

                # Process data in batches
                inserted_count = 0
                for start_idx in range(0, total_rows, batch_size):
                    end_idx = min(start_idx + batch_size, total_rows)
                    batch_df = df[mapped_cols].iloc[start_idx:end_idx]

                    # Prepare data for bulk insert
                    data_tuples = []
                    for _, row in batch_df.iterrows():
                        cleaned_row = []
                        for val in row:
                            if pd.isna(val):
                                cleaned_row.append(None)
                            elif isinstance(val, (int, float, str, bool)):
                                cleaned_row.append(val)
                            else:
                                cleaned_row.append(str(val))
                        data_tuples.append(tuple(cleaned_row))

                    # Use PostgreSQL-specific bulk insert
                    connection = session.connection().connection
                    cursor = connection.cursor()

                    insert_cols = ', '.join(f'"{col}"' for col in postgresql_cols)
                    sql = f'INSERT INTO "{table_name}" ({insert_cols}) VALUES %s'

                    execute_values(cursor, sql, data_tuples, page_size=500)

                    inserted_count += len(data_tuples)
                    if inserted_count % 5000 == 0:
                        info_id(f"Inserted {inserted_count}/{total_rows} rows", self.request_id)

                connection.commit()

                # Analyze table after bulk insert
                session.execute(text(f'ANALYZE "{table_name}"'))
                session.commit()

                info_id(f"Successfully inserted {inserted_count} rows into PostgreSQL table '{table_name}'",
                        self.request_id)
                return inserted_count

            except Exception as e:
                session.rollback()
                error_id(f"Error inserting data into PostgreSQL table '{table_name}': {str(e)}", self.request_id)
                raise

    @with_request_id
    def run(self, sheet_name=None, table_name=None, mapping_name=None, batch_size=1000):
        """Enhanced main execution method with better parameter handling."""
        try:
            # Read Excel file
            with log_timed_operation("read_excel", self.request_id):
                df = self._read_excel_with_validation(sheet_name)

            # Get table and mapping names
            table_name = table_name or self._get_table_name(sheet_name)
            mapping_name = mapping_name or self._get_mapping_name(sheet_name, table_name)

            # Column mapping
            mapping, type_overrides = self.prompt_for_mapping(df)

            if not mapping:
                warning_id("No columns mapped! Exiting.", self.request_id)
                return False

            # Database operations
            info_id("Establishing PostgreSQL database connection", self.request_id)
            session = self.db_config.get_main_session()

            try:
                with log_timed_operation("database_operations", self.request_id):
                    # Create mapping table if needed
                    self.create_mapping_table(session)

                    # Create data table
                    self.create_table(session, table_name, mapping, type_overrides)

                    # Insert data
                    row_count = self.insert_data(session, table_name, df, mapping, batch_size)

                    # Store mapping information with row count
                    self.store_mapping(session, mapping_name, self.excel_path, sheet_name,
                                       table_name, mapping, type_overrides, row_count)

                info_id(f"All PostgreSQL operations completed successfully for mapping '{mapping_name}'",
                        self.request_id)
                return True

            finally:
                debug_id("Closing PostgreSQL database session", self.request_id)
                session.close()

        except Exception as e:
            error_id(f"Error in enhanced Excel to PostgreSQL mapping process: {str(e)}", self.request_id)
            raise

    def _read_excel_with_validation(self, sheet_name):
        """Read Excel file with enhanced validation."""
        try:
            if sheet_name:
                info_id(f"Reading Excel sheet: {sheet_name}", self.request_id)
                df = pd.read_excel(self.excel_path, sheet_name=sheet_name)
            else:
                xls = pd.ExcelFile(self.excel_path)
                info_id(f"Sheets found: {xls.sheet_names}", self.request_id)
                sheet_name = input("Enter sheet name to import: ").strip()
                if sheet_name not in xls.sheet_names:
                    raise ValueError(f"Sheet '{sheet_name}' not found in Excel file")
                df = pd.read_excel(self.excel_path, sheet_name=sheet_name)

            if df.empty:
                raise ValueError(f"Sheet '{sheet_name}' is empty")

            info_id(f"Read {len(df)} rows and {len(df.columns)} columns from sheet '{sheet_name}'", self.request_id)
            return df

        except Exception as e:
            error_id(f"Failed to read Excel file: {e}", self.request_id)
            raise

    def _get_table_name(self, sheet_name):
        """Get table name with validation."""
        default_table = re.sub(r'[^a-zA-Z0-9_]', '_', sheet_name.lower()) if sheet_name else "imported_data"
        table_name = input(f"PostgreSQL table name? (default: {default_table}): ").strip() or default_table
        info_id(f"Using table name: {table_name}", self.request_id)
        return table_name

    def _get_mapping_name(self, sheet_name, table_name):
        """Get mapping name with validation."""
        default_mapping = f"{sheet_name}_to_{table_name}" if sheet_name else f"mapping_to_{table_name}"
        mapping_name = input("Name this mapping (for future use): ").strip() or default_mapping
        info_id(f"Using mapping name: {mapping_name}", self.request_id)
        return mapping_name


# Utility functions for PostgreSQL-specific operations
class PostgreSQLUtilities:
    """Enhanced utility functions specific to PostgreSQL operations."""

    @staticmethod
    def enable_extensions(session, extensions=['pg_trgm', 'unaccent', 'uuid-ossp']):
        """Enable commonly used PostgreSQL extensions with better error handling."""
        try:
            enabled = []
            for ext in extensions:
                try:
                    session.execute(text(f'CREATE EXTENSION IF NOT EXISTS "{ext}"'))
                    enabled.append(ext)
                except Exception as e:
                    debug_id(f"Could not enable extension '{ext}': {e}", get_request_id())
                    continue

            if enabled:
                session.commit()
                info_id(f"Enabled PostgreSQL extensions: {enabled}", get_request_id())
            return enabled
        except Exception as e:
            session.rollback()
            error_id(f"Error enabling PostgreSQL extensions: {str(e)}", get_request_id())
            raise

    @staticmethod
    def create_indexes(session, table_name, columns, index_types=None):
        """Enhanced index creation with different index types."""
        try:
            created_indexes = []
            index_types = index_types or {}

            for column in columns:
                index_name = f"idx_{table_name}_{column}"
                index_type = index_types.get(column, 'btree')

                try:
                    if index_type == 'gin':
                        sql = f'CREATE INDEX IF NOT EXISTS {index_name} ON "{table_name}" USING gin ("{column}")'
                    elif index_type == 'gist':
                        sql = f'CREATE INDEX IF NOT EXISTS {index_name} ON "{table_name}" USING gist ("{column}")'
                    else:
                        sql = f'CREATE INDEX IF NOT EXISTS {index_name} ON "{table_name}" ("{column}")'

                    session.execute(text(sql))
                    created_indexes.append(f"{index_name} ({index_type})")
                except Exception as e:
                    debug_id(f"Could not create index on {column}: {e}", get_request_id())
                    continue

            if created_indexes:
                session.commit()
                info_id(f"Created indexes for table {table_name}: {created_indexes}", get_request_id())
            return created_indexes
        except Exception as e:
            session.rollback()
            error_id(f"Error creating indexes: {str(e)}", get_request_id())
            raise

    @staticmethod
    def analyze_table(session, table_name):
        """Enhanced table analysis with statistics reporting."""
        try:
            session.execute(text(f'ANALYZE "{table_name}"'))

            # Get basic statistics
            stats_query = text("""
                SELECT 
                    schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del, 
                    n_live_tup, n_dead_tup, last_analyze
                FROM pg_stat_user_tables 
                WHERE tablename = :table_name
            """)

            result = session.execute(stats_query, {'table_name': table_name}).fetchone()

            session.commit()
            info_id(f"Analyzed table {table_name} for query optimization", get_request_id())

            if result:
                debug_id(f"Table stats - Live tuples: {result[5]}, Dead tuples: {result[6]}", get_request_id())

            return result
        except Exception as e:
            error_id(f"Error analyzing table {table_name}: {str(e)}", get_request_id())
            raise

    @staticmethod
    def get_table_info(session, table_name):
        """Get comprehensive table information."""
        try:
            info_query = text("""
                SELECT 
                    column_name, data_type, is_nullable, column_default,
                    character_maximum_length, numeric_precision, numeric_scale
                FROM information_schema.columns 
                WHERE table_name = :table_name
                ORDER BY ordinal_position
            """)

            columns = session.execute(info_query, {'table_name': table_name}).fetchall()

            size_query = text("""
                SELECT pg_size_pretty(pg_total_relation_size(:table_name)) as table_size,
                       pg_size_pretty(pg_relation_size(:table_name)) as data_size
            """)

            size_info = session.execute(size_query, {'table_name': table_name}).fetchone()

            return {
                'columns': [dict(zip(['name', 'type', 'nullable', 'default', 'max_length', 'precision', 'scale'], col))
                            for col in columns],
                'total_size': size_info[0] if size_info else 'Unknown',
                'data_size': size_info[1] if size_info else 'Unknown'
            }
        except Exception as e:
            error_id(f"Error getting table info for {table_name}: {str(e)}", get_request_id())
            return None


# ==========================================
# BACKWARD COMPATIBILITY CLASSES
# All use PostgreSQL underneath but keep same names for existing scripts
# ==========================================

class DatabaseManager(PostgreSQLDatabaseManager):
    """
    PostgreSQL-backed database manager - maintains compatibility with existing scripts.
    All operations now use PostgreSQL instead of SQLite.
    """
    pass


class RelationshipManager(PostgreSQLRelationshipManager):
    """
    PostgreSQL-backed relationship manager - maintains compatibility with existing scripts.
    All operations now use PostgreSQL instead of SQLite.
    """

    def associate_parts_with_images_by_title(self, part_ids=None, position_id=None):
        """
        Backward compatible method signature - now uses PostgreSQL with enhanced features.
        """
        return super().associate_parts_with_images_by_title(
            part_ids=part_ids,
            position_id=position_id,
            use_concurrent=True,
            fuzzy_matching=True
        )

    def associate_drawings_with_parts_by_number(self):
        """
        Backward compatible method signature - now uses PostgreSQL with enhanced features.
        """
        return super().associate_drawings_with_parts_by_number(batch_size=100)


class DuplicateManager(PostgreSQLDuplicateManager):
    """
    PostgreSQL-backed duplicate manager - maintains compatibility with existing scripts.
    All operations now use PostgreSQL instead of SQLite.
    """

    def find_duplicate_parts(self, threshold=0.9):
        """
        Backward compatible method signature - now uses PostgreSQL with enhanced features.
        """
        return super().find_duplicate_parts(
            threshold=threshold,
            use_fuzzy_matching=True,
            batch_size=500
        )

    def merge_duplicate_parts(self, source_id, target_id, fields_to_merge=None):
        """
        Backward compatible method signature - now uses PostgreSQL with enhanced features.
        """
        return super().merge_duplicate_parts(
            source_id=source_id,
            target_id=target_id,
            fields_to_merge=fields_to_merge,
            dry_run=False
        )


class EnhancedExcelToSQLiteMapper(EnhancedExcelToPostgreSQLMapper):
    """
    PostgreSQL-backed Excel mapper - maintains compatibility with existing scripts.
    Despite the name, all operations now use PostgreSQL instead of SQLite.
    """

    def __init__(self, excel_path, db_config=None):
        """Initialize with PostgreSQL backend despite SQLite name."""
        super().__init__(excel_path, db_config)
        # Update request ID to reflect PostgreSQL usage
        info_id("Note: EnhancedExcelToSQLiteMapper now uses PostgreSQL backend", self.request_id)

    def infer_sqlite_type(self, pandas_dtype):
        """Backward compatibility method - now returns PostgreSQL types."""
        return self.infer_postgresql_type(pandas_dtype)

    def create_mapping_table(self, session):
        """Creates PostgreSQL mapping table with backward compatible interface."""
        with log_timed_operation("create_mapping_table", self.request_id):
            try:
                # Create PostgreSQL table with SQLite-compatible column names for backward compatibility
                session.execute(text("""
                CREATE TABLE IF NOT EXISTS excel_sqlite_mapping (
                    id SERIAL PRIMARY KEY,
                    mapping_name TEXT UNIQUE,
                    excel_file TEXT NOT NULL,
                    excel_sheet TEXT NOT NULL,
                    sqlite_table TEXT NOT NULL,
                    column_mapping TEXT NOT NULL,
                    column_types TEXT NOT NULL,
                    row_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """))

                session.commit()
                debug_id("PostgreSQL mapping table created (SQLite-compatible schema)", self.request_id)
            except Exception as e:
                session.rollback()
                error_id(f"Error creating mapping table: {str(e)}", self.request_id)
                raise

    def store_mapping(self, session, mapping_name, excel_file, excel_sheet, sqlite_table, mapping, type_overrides,
                      row_count=0):
        """Store mapping with backward compatible parameter names."""
        with log_timed_operation("store_mapping", self.request_id):
            try:
                sql = text("""
                INSERT INTO excel_sqlite_mapping 
                (mapping_name, excel_file, excel_sheet, sqlite_table, column_mapping, column_types, row_count, created_at)
                VALUES (:name, :file, :sheet, :table, :mapping, :types, :row_count, :created_at)
                ON CONFLICT (mapping_name) DO UPDATE SET
                    excel_file = EXCLUDED.excel_file,
                    excel_sheet = EXCLUDED.excel_sheet,
                    sqlite_table = EXCLUDED.sqlite_table,
                    column_mapping = EXCLUDED.column_mapping,
                    column_types = EXCLUDED.column_types,
                    row_count = EXCLUDED.row_count,
                    created_at = EXCLUDED.created_at
                """)

                session.execute(sql, {
                    'name': mapping_name,
                    'file': excel_file,
                    'sheet': excel_sheet,
                    'table': sqlite_table,
                    'mapping': json.dumps(mapping),
                    'types': json.dumps(type_overrides),
                    'row_count': row_count,
                    'created_at': datetime.now()
                })
                session.commit()
                info_id(f"Mapping information stored for '{mapping_name}' (PostgreSQL backend)", self.request_id)
            except Exception as e:
                session.rollback()
                error_id(f"Error storing mapping information: {str(e)}", self.request_id)
                raise


def ensure_database_service_running(env_file='.env', max_wait_seconds=30):
    """
    Ensure PostgreSQL service is running and database is accessible.
    Supports both Windows services and portable PostgreSQL installations.

    Args:
        env_file (str): Path to .env file with database configuration
        max_wait_seconds (int): Maximum time to wait for service to start

    Returns:
        bool: True if database service is running and accessible, False otherwise
    """

    def _log(message, level="INFO"):
        print(f"[{level}] {message}")

    def _is_postgres_running():
        """Check if PostgreSQL process is running - Windows, Linux, macOS compatible"""
        try:
            for proc in psutil.process_iter(['name', 'cmdline']):
                try:
                    proc_name = proc.info['name'].lower()
                    # Check for various PostgreSQL process names
                    postgres_names = ['postgres', 'postgresql', 'pg_ctl', 'postmaster']
                    if any(pg_name in proc_name for pg_name in postgres_names):
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
        except Exception as e:
            _log(f"Error checking PostgreSQL processes: {e}", "WARNING")

        # Fallback: try to connect to the port
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('localhost', 5432))
            sock.close()
            return result == 0
        except Exception:
            pass

        return False

    def _start_postgres():
        """Start PostgreSQL service - handles both Windows services and portable installations"""
        _log("Attempting to start PostgreSQL service...")

        import platform
        system = platform.system().lower()

        if system == "windows":
            # First, try portable PostgreSQL installation (your setup)
            portable_paths = [
                # Your specific paths
                {
                    'bin_dir': r"C:\Users\10169062\Desktop\AU_IndusMaintdb\Database\postgreSQL\pgsql\bin",
                    'data_dir': r"C:\Users\10169062\PostgreSQL\data"
                },
                # Common portable PostgreSQL locations
                {
                    'bin_dir': r"C:\PostgreSQL\bin",
                    'data_dir': r"C:\PostgreSQL\data"
                },
                {
                    'bin_dir': r"C:\Program Files\PostgreSQL\15\bin",
                    'data_dir': r"C:\Program Files\PostgreSQL\15\data"
                },
                {
                    'bin_dir': r"C:\Program Files\PostgreSQL\14\bin",
                    'data_dir': r"C:\Program Files\PostgreSQL\14\data"
                }
            ]

            # Try portable PostgreSQL first
            for paths in portable_paths:
                bin_dir = paths['bin_dir']
                data_dir = paths['data_dir']
                pg_ctl_path = os.path.join(bin_dir, 'pg_ctl.exe')

                if os.path.exists(pg_ctl_path) and os.path.exists(data_dir):
                    try:
                        _log(f"Found portable PostgreSQL: {pg_ctl_path}")
                        _log(f"Using data directory: {data_dir}")

                        # Start PostgreSQL using pg_ctl
                        cmd = [pg_ctl_path, '-D', data_dir, 'start']
                        _log(f"Running command: {' '.join(cmd)}")

                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

                        if result.returncode == 0:
                            _log("PostgreSQL started successfully using pg_ctl")
                            return True
                        elif "already running" in result.stderr.lower() or "already running" in result.stdout.lower():
                            _log("PostgreSQL was already running")
                            return True
                        else:
                            _log(f"pg_ctl failed: {result.stderr}", "WARNING")

                    except Exception as e:
                        _log(f"Error starting PostgreSQL with pg_ctl: {e}", "WARNING")
                        continue

            # If portable PostgreSQL didn't work, try Windows services
            _log("Trying Windows services as fallback...")
            windows_commands = [
                ['net', 'start', 'postgresql-x64-15'],
                ['net', 'start', 'postgresql-x64-14'],
                ['net', 'start', 'postgresql-x64-13'],
                ['net', 'start', 'postgresql-x64-12'],
                ['net', 'start', 'postgresql'],
                ['sc', 'start', 'postgresql-x64-15'],
                ['sc', 'start', 'postgresql-x64-14'],
                ['net', 'start', 'PostgreSQL'],
            ]

            for cmd in windows_commands:
                try:
                    _log(f"Trying Windows command: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, shell=True)
                    if result.returncode == 0:
                        _log(f"PostgreSQL started using: {' '.join(cmd)}")
                        return True
                    elif "already" in result.stdout.lower() or "already" in result.stderr.lower():
                        _log("PostgreSQL service was already running")
                        return True
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
                    _log(f"Command failed: {e}", "DEBUG")
                    continue

            # If all attempts fail, provide helpful advice
            _log("Could not start PostgreSQL on Windows", "ERROR")
            _log("💡 Manual steps to try:", "INFO")
            _log("   1. Run this command manually:", "INFO")
            _log('      cd "C:\\Users\\10169062\\Desktop\\AU_IndusMaintdb\\Database\\postgreSQL\\pgsql\\bin"', "INFO")
            _log('      .\\pg_ctl.exe -D "C:\\Users\\10169062\\PostgreSQL\\data" start', "INFO")
            _log("   2. Or check Windows Services (services.msc)", "INFO")
            _log("   3. Or verify PostgreSQL is installed correctly", "INFO")
            return False

        else:
            # Linux and macOS service management (unchanged)
            unix_commands = [
                ['sudo', 'systemctl', 'start', 'postgresql'],
                ['brew', 'services', 'start', 'postgresql'],
                ['sudo', 'service', 'postgresql', 'start'],
                ['sudo', 'brew', 'services', 'start', 'postgresql@14'],
                ['sudo', 'brew', 'services', 'start', 'postgresql@13'],
            ]

            for cmd in unix_commands:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    if result.returncode == 0:
                        _log(f"PostgreSQL started using: {' '.join(cmd)}")
                        return True
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                    continue

            _log("Could not start PostgreSQL - no suitable service manager found", "ERROR")
            return False

    def _test_database_connection():
        """Test actual database connectivity using environment variables"""
        try:
            import psycopg2
        except ImportError:
            _log("psycopg2 not available for connection testing", "WARNING")
            return True  # Assume it's okay if we can't test

        try:
            # Get connection parameters from environment
            host = os.getenv('POSTGRES_HOST', 'localhost')
            port = os.getenv('POSTGRES_PORT', '5432')
            user = os.getenv('POSTGRES_USER', 'postgres')
            password = os.getenv('POSTGRES_PASSWORD', '')
            database = os.getenv('POSTGRES_DB', 'postgres')

            _log("Testing database connection...")

            # Test connection with timeout
            conn = psycopg2.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                connect_timeout=10
            )

            # Quick test query
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()

            cursor.close()
            conn.close()

            _log("Database connection test successful")
            return True

        except Exception as e:
            _log(f"Database connection test failed: {str(e)}", "ERROR")
            return False

    # Main logic
    _log("Checking PostgreSQL database service...")

    # Load environment variables
    if os.path.exists(env_file):
        load_dotenv(env_file)
        _log(f"Using absolute .env path: {os.path.abspath(env_file)}", "DEBUG")
        _log(f"Loaded environment variables from {env_file}")
        # -----------------------------------------------------
        # 🔍 Debug: Show the key environment variables in use
        # -----------------------------------------------------
        env_debug = {
            "POSTGRES_USER": os.getenv("POSTGRES_USER"),
            "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD"),
            "POSTGRES_HOST": os.getenv("POSTGRES_HOST"),
            "POSTGRES_PORT": os.getenv("POSTGRES_PORT"),
            "POSTGRES_DB": os.getenv("POSTGRES_DB"),
            "DATABASE_URL": os.getenv("DATABASE_URL"),
            "USE_AZURE_DB": os.getenv("USE_AZURE_DB"),
        }

        _log("Loaded environment values:", "DEBUG")
        for k, v in env_debug.items():
            masked_value = v if (k not in ["POSTGRES_PASSWORD"] and v) else "***MASKED***"
            _log(f"   {k} = {masked_value}", "DEBUG")

    else:
        _log(f"Environment file {env_file} not found, using system environment", "WARNING")

    # Check if PostgreSQL is already running
    if _is_postgres_running():
        _log("PostgreSQL service is already running")
    else:
        _log("PostgreSQL service is not running")

        if not _start_postgres():
            _log("Failed to start PostgreSQL service", "ERROR")

            # Provide platform-specific troubleshooting advice
            import platform
            if platform.system().lower() == "windows":
                _log("🔧 Windows PostgreSQL Troubleshooting:", "INFO")
                _log("   1. Try starting manually:", "INFO")
                _log('      cd "C:\\Users\\10169062\\Desktop\\AU_IndusMaintdb\\Database\\postgreSQL\\pgsql\\bin"',
                     "INFO")
                _log('      .\\pg_ctl.exe -D "C:\\Users\\10169062\\PostgreSQL\\data" start', "INFO")
                _log("   2. Check if data directory exists:", "INFO")
                _log('      C:\\Users\\10169062\\PostgreSQL\\data', "INFO")
                _log("   3. Verify PostgreSQL installation", "INFO")
                _log("   4. Check .env file database credentials", "INFO")
            else:
                _log("🔧 Unix/Linux PostgreSQL Troubleshooting:", "INFO")
                _log("   1. Install PostgreSQL: sudo apt install postgresql", "INFO")
                _log("   2. Start manually: sudo systemctl start postgresql", "INFO")
                _log("   3. Enable auto-start: sudo systemctl enable postgresql", "INFO")

            return False

        # Wait for service to fully start
        _log("⏳ Waiting for PostgreSQL service to fully initialize...")
        start_time = time.time()

        while time.time() - start_time < max_wait_seconds:
            if _is_postgres_running():
                _log("PostgreSQL service started successfully")
                break
            time.sleep(1)
        else:
            _log("PostgreSQL service failed to start within timeout", "ERROR")
            return False

    # Test database connectivity
    if _test_database_connection():
        _log("Database service is ready for connections")
        return True
    else:
        _log("PostgreSQL service is running but database connection failed", "WARNING")
        _log("🔧 Connection Troubleshooting:", "INFO")
        _log("   1. Check your .env file database credentials", "INFO")
        _log("   2. Verify database exists (create it if needed)", "INFO")
        _log("   3. Check PostgreSQL authentication settings (pg_hba.conf)", "INFO")
        _log("   4. Ensure user has proper permissions", "INFO")
        return False


# Convenience function for use with your DatabaseManager
def get_database_manager_safely(**kwargs):
    """
    Get a DatabaseManager instance, ensuring the database service is running first.

    Args:
        **kwargs: Arguments to pass to DatabaseManager constructor

    Returns:
        DatabaseManager: Ready-to-use database manager instance

    Raises:
        RuntimeError: If database service cannot be started or is not accessible
    """
    if not ensure_database_service_running():
        raise RuntimeError("Database service is not available")

    return DatabaseManager(**kwargs)