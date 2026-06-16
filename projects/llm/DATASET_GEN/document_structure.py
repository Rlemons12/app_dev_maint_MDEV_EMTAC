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