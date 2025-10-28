from sqlalchemy import Column, Integer, String, ForeignKey, Table, LargeBinary
from sqlalchemy.orm import declarative_base, relationship

# Base class for ORM models
Base = declarative_base()

# Association Table for Many-to-Many Relationship between Principles and SOPs
principle_sop_association = Table(
    'principle_sop_association',
    Base.metadata,
    Column('principal_id', Integer, ForeignKey('principal_of_operation.id')),
    Column('sop_id', Integer, ForeignKey('standard_operation_procedures.id'))
)


class MachineComponent(Base):
    __tablename__ = 'machine_component_structure_diagram'

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey('image.id'))

    # Relationship to PrincipalOfOperation
    operations = relationship('PrincipalOfOperation', back_populates='machine_component')

    def __repr__(self):
        return f"<MachineComponent(id={self.id}, image_id={self.image_id})>"


class PrincipalOfOperation(Base):
    __tablename__ = 'principal_of_operation'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    position_id = Column(Integer, ForeignKey('position.id'))
    ideal_sequence_of_movement = Column(String, nullable=False)
    required_conditions_of_each_ideal_movement = Column(String, nullable=False)

    machine_component_structure_diagram_id = Column(Integer, ForeignKey('machine_component_structure_diagram.id'))

    # Relationship to MachineComponent
    machine_component = relationship('MachineComponent', back_populates='operations')

    # Many-to-Many relationship with SOPs
    related_sops = relationship(
        'StandardOperationProcedures',
        secondary=principle_sop_association,
        back_populates='related_principles'
    )

    def __repr__(self):
        return (f"<PrincipalOfOperation(id={self.id}, name='{self.name}', "
                f"ideal_sequence_of_movement='{self.ideal_sequence_of_movement}', "
                f"required_conditions_of_each_ideal_movement='{self.required_conditions_of_each_ideal_movement}')>")


class StandardOperationProcedures(Base):
    __tablename__ = 'standard_operation_procedures'

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_number = Column(String, nullable=False)
    description = Column(String, nullable=False)
    purpose = Column(String, nullable=False)
    scope = Column(String, nullable=False)
    references = Column(String, nullable=False)

    position_id = Column(Integer, ForeignKey('position.id'))

    # Many-to-Many relationship with Principles of Operation
    related_principles = relationship(
        'PrincipalOfOperation',
        secondary=principle_sop_association,
        back_populates='related_sops'
    )

    # One-to-One relationship with SOP Embedding
    embedding = relationship('SOPEmbedding', uselist=False, back_populates='sop')

    def __repr__(self):
        return (f"<StandardOperationProcedures(id={self.id}, document_number='{self.document_number}', "
                f"description='{self.description}', purpose='{self.purpose}', "
                f"scope='{self.scope}', references='{self.references}')>")


class SOPEmbedding(Base):
    __tablename__ = 'sop_embeddings'

    id = Column(Integer, primary_key=True, autoincrement=True)
    sop_id = Column(Integer, ForeignKey('standard_operation_procedures.id'), nullable=False)

    # Storing the embedding securely
    encrypted_embedding_vector = Column(LargeBinary, nullable=False)

    # Relationship back to SOP
    sop = relationship('StandardOperationProcedures', back_populates='embedding')

    def __repr__(self):
        return f"<SOPEmbedding(id={self.id}, sop_id={self.sop_id})>"
