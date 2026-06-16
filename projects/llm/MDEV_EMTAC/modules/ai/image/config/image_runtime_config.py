"""Runtime image module configuration and preserved globals."""


from __future__ import annotations


import logging

from modules.configuration.config import DATABASE_URL, ALLOWED_EXTENSIONS
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


engine = create_engine(DATABASE_URL)


Session = sessionmaker(bind=engine)
