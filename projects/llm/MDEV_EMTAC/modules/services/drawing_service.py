# services/drawing_service.py

from __future__ import annotations

import os
import shutil
from pathlib import Path, PureWindowsPath
from typing import Optional, List, Dict, Any

from sqlalchemy.exc import SQLAlchemyError

from modules.emtacdb.emtacdb_fts import Drawing
from modules.configuration.config import DATABASE_DRAWING, DATABASE_DIR
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import info_id, error_id, with_request_id


class DrawingService:
    """
    Service layer for managing Drawing entities.

    Handles:
      - create drawing records
      - update drawing records
      - delete drawing records
      - optionally delete physical drawing files
      - copy drawing files into the configured drawing folder
      - store Drawing.file_path in the same style as images

    IMPORTANT FILE PATH RULE:

        DATABASE_DIR:
            E:\\emtac\\Database

        DATABASE_DRAWING:
            E:\\emtac\\Database\\DB_DRAWING

        Physical drawing file:
            E:\\emtac\\Database\\DB_DRAWING\\AFL31600\\E-5347-2-2.dwg

        Database Drawing.file_path value:
            DB_DRAWING\\AFL31600\\E-5347-2-2.dwg

    This matches the image storage style:

        DB_IMAGES\\some_image.jpeg

    The service can still resolve database-relative paths back to physical files
    when it needs to delete or inspect files.
    """

    def __init__(self, db_config: Optional[DatabaseConfig] = None):
        self.db_config = db_config or DatabaseConfig()

    # ---------------------------------------------------------------------
    # PATH HELPERS
    # ---------------------------------------------------------------------

    def get_database_root(self) -> Path:
        """
        Return the main database folder.

        Example:
            E:\\emtac\\Database
        """
        if DATABASE_DIR and str(DATABASE_DIR).strip():
            database_root = Path(str(DATABASE_DIR)).expanduser().resolve()
        else:
            # Fallback if DATABASE_DIR is not available for some reason.
            database_root = self.get_drawing_root().parent.resolve()

        database_root.mkdir(parents=True, exist_ok=True)
        return database_root

    def get_drawing_root(self) -> Path:
        """
        Return the physical folder where drawing files are stored.

        Example:
            E:\\emtac\\Database\\DB_DRAWING
        """
        if not DATABASE_DRAWING or not str(DATABASE_DRAWING).strip():
            raise ValueError(
                "DATABASE_DRAWING is not configured. "
                "Add this to your .env file. Example: "
                "DATABASE_DRAWING=E:\\emtac\\Database\\DB_DRAWING"
            )

        drawing_root = Path(str(DATABASE_DRAWING)).expanduser().resolve()
        drawing_root.mkdir(parents=True, exist_ok=True)

        return drawing_root

    @staticmethod
    def _path_text(value: str) -> str:
        """
        Normalize a path string without resolving it to absolute.

        This keeps database values consistent.
        """
        return os.path.normpath(str(value).strip())

    @staticmethod
    def _is_absolute_path(value: str) -> bool:
        """
        Detect absolute paths.

        Uses both Path and PureWindowsPath so Windows paths like E:\\folder\\file.dwg
        are recognized even if inspected from a non-Windows environment.
        """
        text = str(value).strip()

        if not text:
            return False

        return Path(text).expanduser().is_absolute() or PureWindowsPath(text).is_absolute()

    def _first_path_part(self, file_path: str) -> Optional[str]:
        """
        Return the first part of a path.

        Example:
            DB_DRAWING\\AFL31600\\E-5347-2-2.dwg

        Returns:
            DB_DRAWING
        """
        text = self._path_text(file_path)

        if not text:
            return None

        path = Path(text)

        if path.parts:
            return path.parts[0]

        # Fallback for odd slash/backslash input.
        normalized = text.replace("/", "\\")
        parts = [part for part in normalized.split("\\") if part]

        if parts:
            return parts[0]

        return None

    def resolve_physical_file_path(self, file_path: str) -> Path:
        """
        Resolve a database or user-provided drawing path into a physical file path.

        Supports these inputs:

            Absolute:
                E:\\emtac\\Database\\DB_DRAWING\\AFL31600\\E-5347-2-2.dwg

            DATABASE_DIR-relative:
                DB_DRAWING\\AFL31600\\E-5347-2-2.dwg

            DATABASE_DRAWING-relative:
                AFL31600\\E-5347-2-2.dwg

            Filename only:
                E-5347-2-2.dwg

        Preferred database storage format:

            DB_DRAWING\\AFL31600\\E-5347-2-2.dwg
        """
        if file_path is None or not str(file_path).strip():
            raise ValueError("file_path is required.")

        text = self._path_text(file_path)

        if self._is_absolute_path(text):
            return Path(text).expanduser().resolve()

        database_root = self.get_database_root()
        drawing_root = self.get_drawing_root()
        drawing_folder_name = drawing_root.name.lower()

        first_part = self._first_path_part(text)

        if first_part and first_part.lower() == drawing_folder_name:
            return (database_root / text).resolve()

        return (drawing_root / text).resolve()

    def to_database_relative_file_path(self, physical_file_path: str | Path) -> str:
        """
        Convert a physical drawing file path into a DATABASE_DIR-relative path.

        Example:
            physical_file_path:
                E:\\emtac\\Database\\DB_DRAWING\\AFL31600\\E-5347-2-2.dwg

            returns:
                DB_DRAWING\\AFL31600\\E-5347-2-2.dwg
        """
        resolved_file = Path(physical_file_path).expanduser().resolve()
        drawing_root = self.get_drawing_root()
        database_root = self.get_database_root()

        try:
            resolved_file.relative_to(drawing_root)
        except ValueError:
            raise ValueError(
                "Drawing file must be inside DATABASE_DRAWING before it can be "
                "stored as a managed drawing path. "
                f"file_path={resolved_file}, drawing_root={drawing_root}"
            )

        try:
            relative_path = resolved_file.relative_to(database_root)
        except ValueError:
            # Fallback if DATABASE_DRAWING is not physically below DATABASE_DIR.
            relative_to_drawing = resolved_file.relative_to(drawing_root)
            relative_path = Path(drawing_root.name) / relative_to_drawing

        return os.path.normpath(str(relative_path))

    def normalize_file_path(self, file_path: str) -> str:
        """
        Normalize a drawing file path for database storage.

        This method returns the value that should be stored in Drawing.file_path.

        Preferred output:
            DB_DRAWING\\AFL31600\\E-5347-2-2.dwg

        It does NOT return:
            E:\\emtac\\Database\\DB_DRAWING\\AFL31600\\E-5347-2-2.dwg
        """
        physical_path = self.resolve_physical_file_path(file_path)
        return self.to_database_relative_file_path(physical_path)

    def copy_file_to_drawing_folder(
        self,
        source_file_path: str,
        target_filename: Optional[str] = None,
        overwrite: bool = False,
    ) -> str:
        """
        Copy a file into DATABASE_DRAWING and return the database storage path.

        Returned value example:
            DB_DRAWING\\AFL31600\\E-5347-2-2.dwg

        If target_filename includes a subfolder, that subfolder is created under
        DATABASE_DRAWING.

        Examples:
            target_filename="E-5347-2-2.dwg"
            target_filename="AFL31600\\E-5347-2-2.dwg"
        """
        if not source_file_path or not str(source_file_path).strip():
            raise ValueError("source_file_path is required.")

        source_path = Path(str(source_file_path)).expanduser().resolve()

        if not source_path.exists():
            raise FileNotFoundError(f"Drawing source file does not exist: {source_path}")

        if not source_path.is_file():
            raise ValueError(f"Drawing source path is not a file: {source_path}")

        drawing_root = self.get_drawing_root()

        final_target = target_filename or source_path.name
        final_target = self._path_text(final_target)

        # If someone passes DB_DRAWING\\file.dwg as target_filename,
        # strip the leading DB_DRAWING because destination is already inside that folder.
        first_part = self._first_path_part(final_target)
        if first_part and first_part.lower() == drawing_root.name.lower():
            parts = Path(final_target).parts
            final_target = os.path.normpath(str(Path(*parts[1:]))) if len(parts) > 1 else source_path.name

        destination_path = (drawing_root / final_target).resolve()
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        if destination_path.exists() and not overwrite:
            raise FileExistsError(
                f"Drawing file already exists and overwrite=False: {destination_path}"
            )

        shutil.copy2(source_path, destination_path)

        return self.to_database_relative_file_path(destination_path)

    def resolve_stored_file_path(self, stored_file_path: Optional[str]) -> Optional[Path]:
        """
        Resolve a Drawing.file_path value from the database to a physical path.

        Example:
            DB_DRAWING\\AFL31600\\E-5347-2-2.dwg

        resolves to:
            E:\\emtac\\Database\\DB_DRAWING\\AFL31600\\E-5347-2-2.dwg
        """
        if stored_file_path is None or not str(stored_file_path).strip():
            return None

        return self.resolve_physical_file_path(stored_file_path)

    # ---------------------------------------------------------------------
    # FORMAT HELPER
    # ---------------------------------------------------------------------

    def to_dict(self, drawing: Drawing) -> Dict[str, Any]:
        """
        Convert a Drawing ORM object into a dictionary safe for API responses.
        """
        return {
            "id": drawing.id,
            "drw_equipment_name": drawing.drw_equipment_name,
            "drw_number": drawing.drw_number,
            "drw_name": drawing.drw_name,
            "drw_revision": drawing.drw_revision,
            "drw_spare_part_number": drawing.drw_spare_part_number,
            "drw_type": drawing.drw_type,
            "file_path": drawing.file_path,
            "url": f"/drawings/view/{drawing.id}",
        }

    # ---------------------------------------------------------------------
    # CORE CRUD
    # ---------------------------------------------------------------------

    @with_request_id
    def add_to_db(
        self,
        drw_equipment_name: Optional[str],
        drw_number: str,
        file_path: str,
        drw_name: Optional[str] = None,
        drw_revision: Optional[str] = None,
        drw_spare_part_number: Optional[str] = None,
        drw_type: str = "Other",
        request_id: Optional[str] = None,
    ) -> int:
        """
        Create a new Drawing and return its ID.

        file_path may be:
            - absolute path under DATABASE_DRAWING
            - DB_DRAWING-relative path
            - DATABASE_DRAWING-relative path

        Stored value will be:
            DB_DRAWING\\...
        """
        if not drw_number or not str(drw_number).strip():
            raise ValueError("drw_number is required.")

        database_file_path = self.normalize_file_path(file_path)

        with self.db_config.main_session() as session:
            try:
                drawing = Drawing(
                    drw_equipment_name=drw_equipment_name,
                    drw_number=str(drw_number).strip(),
                    drw_name=drw_name,
                    drw_revision=drw_revision,
                    drw_spare_part_number=drw_spare_part_number,
                    drw_type=drw_type or "Other",
                    file_path=database_file_path,
                )

                session.add(drawing)
                session.commit()
                session.refresh(drawing)

                info_id(
                    (
                        f"Created Drawing '{drawing.drw_number}' "
                        f"id={drawing.id}, file_path={database_file_path}"
                    ),
                    request_id,
                )

                return drawing.id

            except SQLAlchemyError as e:
                session.rollback()
                error_id(f"DrawingService.add_to_db failed: {e}", request_id)
                raise

    @with_request_id
    def create_from_source_file(
        self,
        source_file_path: str,
        drw_equipment_name: Optional[str],
        drw_number: str,
        drw_name: Optional[str] = None,
        drw_revision: Optional[str] = None,
        drw_spare_part_number: Optional[str] = None,
        drw_type: str = "Other",
        target_filename: Optional[str] = None,
        overwrite_file: bool = False,
        request_id: Optional[str] = None,
    ) -> int:
        """
        Copy a physical file into DATABASE_DRAWING, then create the Drawing record.

        The database file_path will be stored as:
            DB_DRAWING\\...
        """
        database_file_path = self.copy_file_to_drawing_folder(
            source_file_path=source_file_path,
            target_filename=target_filename,
            overwrite=overwrite_file,
        )

        return self.add_to_db(
            drw_equipment_name=drw_equipment_name,
            drw_number=drw_number,
            drw_name=drw_name,
            drw_revision=drw_revision,
            drw_spare_part_number=drw_spare_part_number,
            drw_type=drw_type,
            file_path=database_file_path,
            request_id=request_id,
        )

    @with_request_id
    def get(
        self,
        drawing_id: int,
        as_dict: bool = False,
        request_id: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Retrieve a Drawing by ID.
        """
        with self.db_config.main_session() as session:
            try:
                drawing = session.query(Drawing).filter(Drawing.id == drawing_id).first()

                if not drawing:
                    return None

                if as_dict:
                    return self.to_dict(drawing)

                return drawing

            except SQLAlchemyError as e:
                error_id(f"DrawingService.get failed for id={drawing_id}: {e}", request_id)
                raise

    @with_request_id
    def update(
        self,
        drawing_id: int,
        drw_equipment_name: Optional[str] = None,
        drw_number: Optional[str] = None,
        drw_name: Optional[str] = None,
        drw_revision: Optional[str] = None,
        drw_spare_part_number: Optional[str] = None,
        drw_type: Optional[str] = None,
        file_path: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Update a Drawing record.

        Only provided values are updated.

        If file_path is provided, it will be stored as:
            DB_DRAWING\\...
        """
        with self.db_config.main_session() as session:
            try:
                drawing = session.query(Drawing).filter(Drawing.id == drawing_id).first()

                if not drawing:
                    return None

                if drw_equipment_name is not None:
                    drawing.drw_equipment_name = drw_equipment_name

                if drw_number is not None:
                    if not str(drw_number).strip():
                        raise ValueError("drw_number cannot be blank.")
                    drawing.drw_number = str(drw_number).strip()

                if drw_name is not None:
                    drawing.drw_name = drw_name

                if drw_revision is not None:
                    drawing.drw_revision = drw_revision

                if drw_spare_part_number is not None:
                    drawing.drw_spare_part_number = drw_spare_part_number

                if drw_type is not None:
                    drawing.drw_type = drw_type or "Other"

                if file_path is not None:
                    drawing.file_path = self.normalize_file_path(file_path)

                session.commit()
                session.refresh(drawing)

                info_id(f"Updated Drawing id={drawing_id}", request_id)

                return self.to_dict(drawing)

            except SQLAlchemyError as e:
                session.rollback()
                error_id(f"DrawingService.update failed for id={drawing_id}: {e}", request_id)
                raise

    @with_request_id
    def update_from_source_file(
        self,
        drawing_id: int,
        source_file_path: str,
        target_filename: Optional[str] = None,
        overwrite_file: bool = False,
        request_id: Optional[str] = None,
        **drawing_updates,
    ) -> Optional[Dict[str, Any]]:
        """
        Copy a new physical file into DATABASE_DRAWING, then update the drawing record.

        The database file_path will be stored as:
            DB_DRAWING\\...
        """
        database_file_path = self.copy_file_to_drawing_folder(
            source_file_path=source_file_path,
            target_filename=target_filename,
            overwrite=overwrite_file,
        )

        drawing_updates["file_path"] = database_file_path

        return self.update(
            drawing_id=drawing_id,
            request_id=request_id,
            **drawing_updates,
        )

    @with_request_id
    def remove(
        self,
        drawing_id: int,
        delete_file: bool = False,
        request_id: Optional[str] = None,
    ) -> bool:
        """
        Delete a Drawing record by ID.

        delete_file=False:
            Deletes only the database record.

        delete_file=True:
            Deletes the database record and attempts to delete the physical file.

        Supports stored file_path values like:
            DB_DRAWING\\AFL31600\\E-5347-2-2.dwg
        """
        with self.db_config.main_session() as session:
            try:
                drawing = session.query(Drawing).filter(Drawing.id == drawing_id).first()

                if not drawing:
                    return False

                stored_file_path = drawing.file_path
                physical_path = self.resolve_stored_file_path(stored_file_path)

                session.delete(drawing)
                session.commit()

                if delete_file and physical_path:
                    if physical_path.exists() and physical_path.is_file():
                        physical_path.unlink()
                        info_id(
                            f"Deleted physical drawing file: {physical_path}",
                            request_id,
                        )

                info_id(f"Deleted Drawing id={drawing_id}", request_id)

                return True

            except SQLAlchemyError as e:
                session.rollback()
                error_id(f"DrawingService.remove failed for id={drawing_id}: {e}", request_id)
                raise

    def delete(
        self,
        drawing_id: int,
        delete_file: bool = False,
        request_id: Optional[str] = None,
    ) -> bool:
        """
        Alias for remove().
        """
        return self.remove(
            drawing_id=drawing_id,
            delete_file=delete_file,
            request_id=request_id,
        )

    # ---------------------------------------------------------------------
    # SEARCH HELPERS
    # ---------------------------------------------------------------------

    @with_request_id
    def find(
        self,
        request_id: Optional[str] = None,
        **filters,
    ) -> List[Drawing]:
        """
        Search for Drawings.

        This wraps Drawing.search(), which already supports:
          - search_text
          - fields
          - exact_match
          - drawing_id
          - drw_equipment_name
          - drw_number
          - drw_name
          - drw_revision
          - drw_spare_part_number
          - drw_type
          - file_path
          - limit
        """
        try:
            return Drawing.search(
                request_id=request_id,
                **filters,
            )
        except SQLAlchemyError as e:
            error_id(f"DrawingService.find failed: {e}", request_id)
            raise

    @with_request_id
    def find_formatted(
        self,
        request_id: Optional[str] = None,
        **filters,
    ) -> Dict[str, Any]:
        """
        Search and return formatted drawing results.
        """
        try:
            return Drawing.search_and_format(
                request_id=request_id,
                **filters,
            )
        except SQLAlchemyError as e:
            error_id(f"DrawingService.find_formatted failed: {e}", request_id)
            raise

    @with_request_id
    def find_by_asset(
        self,
        asset_number: str,
        request_id: Optional[str] = None,
    ) -> List[Drawing]:
        """
        Search drawings by asset number.
        """
        try:
            return Drawing.search_by_asset_number(
                asset_number,
                request_id=request_id,
            )
        except SQLAlchemyError as e:
            error_id(f"DrawingService.find_by_asset failed: {e}", request_id)
            raise

    @with_request_id
    def find_by_type(
        self,
        drawing_type: str,
        request_id: Optional[str] = None,
    ) -> List[Drawing]:
        """
        Search drawings by drawing type.
        """
        try:
            return Drawing.search_by_type(
                drawing_type,
                request_id=request_id,
            )
        except SQLAlchemyError as e:
            error_id(f"DrawingService.find_by_type failed: {e}", request_id)
            raise

    @with_request_id
    def available_types(
        self,
        request_id: Optional[str] = None,
    ) -> List[str]:
        """
        Return all available drawing types.
        """
        return Drawing.get_available_types()

    # ---------------------------------------------------------------------
    # RELATIONSHIPS
    # ---------------------------------------------------------------------

    @with_request_id
    def find_related(
        self,
        drawing_id: int,
        request_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Return related entities for a Drawing.

        Includes:
          - positions
          - problems
          - tasks
          - parts
        """
        with self.db_config.main_session() as session:
            try:
                drawing = session.query(Drawing).filter(Drawing.id == drawing_id).first()

                if not drawing:
                    return None

                return {
                    "drawing": self.to_dict(drawing),
                    "downward": {
                        "positions": drawing.drawing_position,
                        "problems": drawing.drawing_problem,
                        "tasks": drawing.drawing_task,
                        "parts": drawing.drawing_part,
                    },
                }

            except SQLAlchemyError as e:
                error_id(
                    f"DrawingService.find_related failed for id={drawing_id}: {e}",
                    request_id,
                )
                raise