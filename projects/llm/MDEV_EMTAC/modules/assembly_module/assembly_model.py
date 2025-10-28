import sys
import os
from sqlalchemy.orm import relationship
from sqlalchemy import (Column, ForeignKey, Integer, String)
if getattr(sys, 'frozen', False):  # Check if running as an executable
    current_dir = os.path.dirname(sys.executable)  # Use the executable directory
else:
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Use the script directory
from modules.configuration.base import Base


sys.path.append(current_dir)
class Assembly(Base):
    #main assembly of a location
    __tablename__ = 'assembly'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)

    # Relationships
    subassemblies = relationship("ComponentAssembly", back_populates="assembly")
    
class SubAssembly(Base):
    #specific group of components of an assembly
    __tablename__ = 'subassembly'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    assembly_id = Column(Integer, ForeignKey('assembly.id'), nullable=False)
    assembly_view_id = Column(Integer, ForeignKey('assembly_view.id'), nullable=False)

    # Relationships
    assembly = relationship("Subassembly", back_populates="subassemblies")
    assembly_view = relationship("AssemblyView", back_populates="subassemblies")

class AssemblyView(Base):
    __tablename__ = 'assembly_view'
    #location within assembly. example front,back,right-side top left ect...
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)

    # Relationships
    subassemblies = relationship("ComponentAssembly", back_populates="assembly_view")








