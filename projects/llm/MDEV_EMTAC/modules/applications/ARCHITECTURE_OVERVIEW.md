# EMTAC Architecture Overview

**Adapters → Coordinator → Orchestrator → Service → Database Framework**

---

## 1. Purpose

This document defines the layered architecture used in the EMTAC system.

The goal is to enforce:

- Clear separation of concerns  
- Transaction safety  
- Testability  
- Offline-first reliability  
- Enterprise-grade maintainability  
- Predictable data flow  

This structure supports ingestion pipelines, AI model integration, document processing, and domain logic **without cross-layer contamination**.

---

## 2. High-Level Architecture


UI (Web / CLI / Job / API)
↓
Transport Layer (Adapters, Routes)
↓
Application Coordinators
↓
Domain Orchestrators
↓
Domain Services
↓
Database Framework (ORM / Persistence)
↓
PostgreSQL / pgvector


Each layer has **strict responsibilities** and **hard boundaries**.

---

## 3. Layer Definitions

---

### 3.1 Transport Layer

**Code Location**

Flask routes
CLI scripts
Background jobs
GPU service endpoints
modules/adapters


**Examples**

- Flask upload routes  
- CLI ingestion scripts  
- `LocalFileAdapter`  
- Future: `S3FileAdapter`, `RemoteFileAdapter`

**Responsibilities**

- Receive user input  
- Validate basic structure  
- Convert external input → application contract  
- Call an Application Coordinator  
- Return formatted responses  

**Rules**

Transport layer **must never**:

- Open database sessions  
- Commit or rollback transactions  
- Contain business logic  
- Call ORM models directly  
- Call services directly  

The transport layer **always calls a Coordinator**.

---

### 3.2 Application Coordinator Layer

**Code Location**

modules/applications/


**Example**

- `FileProcessingCoordinator`

**Responsibilities**

- Validate high-level inputs  
- Detect file types or ingestion modes  
- Route processing to appropriate orchestrator  
- Aggregate multi-step results  
- Normalize response format  
- Remain transaction-agnostic  

**What It Does NOT Do**

- No session management  
- No commit / rollback  
- No ORM usage  
- No direct SQL  

**Why It Exists**

It prevents route and transport logic from becoming complex and ensures workflows remain clean and reusable.

The Coordinator is the **entry point into the application core**.

---

### 3.3 Orchestrator Layer

**Code Location**

modules/orchestrators/


**Examples**

- `CompleteDocumentOrchestrator`  
- `ImageOrchestrator`  
- `AIModelOrchestrator`  
- `ToolOrchestrator`  

**Responsibilities**

- Own session lifecycle  
- Own transaction boundaries  
- Coordinate multiple services  
- Enforce business rules  
- Maintain logging and request IDs  
- Roll back on failure  

**Example Transaction Pattern**

```python
with self.transaction() as session:
    entity = self.services.document.save(session, ...)
    self.services.embedding.generate(session, ...)

Rules

✔ May open sessions
✔ May commit
✔ May rollback
✔ May coordinate multiple services

❌ Must not contain SQL
❌ Must not directly manipulate tables
❌ Must not bypass services

Why Orchestrators Exist

They are the transaction guardians.

They prevent:

Session leaks

Partial commits

Cross-service coupling

Business logic living in routes

3.4 Service Layer

Code Location

modules/services/

Examples

DocumentService

ImageService

EmbeddingService

FileStorageService

ContentExtractionService

Responsibilities

CRUD operations

Query composition

Validation logic

Domain-level calculations

Hard Rules

Services must:

NEVER open sessions

NEVER commit

NEVER rollback

NEVER control transactions

They receive a session only from the orchestrator.

Example

def save(self, session: Session, *, name: str):
    doc = Document(name=name)
    session.add(doc)
    return doc

Why Services Exist

They isolate database logic from orchestration logic, making:

Unit testing easy

Refactoring safe

Business rules reusable

3.5 Database Framework Layer

Code Location

modules/emtacdb/
modules/configuration/

Includes

SQLAlchemy ORM models

DatabaseConfig

Session factory

pgvector integration

Migration layer

Responsibilities

Define schema

Define relationships

Define indexes

Define constraints

Provide session factory

What It Does NOT Do

No business logic

No cross-table workflows

No orchestration

No API awareness

4. Data Flow Example
Example: PDF Upload
Flask Route / CLI Script
        ↓
LocalFileAdapter (if needed)
        ↓
FileProcessingCoordinator
        ↓
CompleteDocumentOrchestrator
        ↓
DocumentService.save()
        ↓
EmbeddingService.generate()
        ↓
ImageService.extract()
        ↓
Commit Transaction

Single transaction

Single commit

Rollback on any failure

5. Transaction Philosophy

Only Orchestrators may:

Open sessions

Commit

Rollback

Control transaction boundaries

This guarantees:

No partial data writes

No inconsistent embeddings

No broken associations

Clean failure recovery

6. Why This Architecture Is Critical for EMTAC

Your system includes:

AI extraction

Embedding generation

Chunk splitting

Image extraction

pgvector similarity search

Multi-model backends (local + GPU)

Offline-first processing

Without strict layering:

Sessions would leak

Embeddings would mismatch

Documents would partially save

Rollbacks would fail

Services would become stateful

This architecture prevents that.

7. Architectural Principles Enforced
Principle	How It Is Enforced
Separation of concerns	Strict layer and folder boundaries
Transaction integrity	Orchestrator ownership
Stateless services	No session control in services
Testability	Services testable with injected session
Scalability	Coordinators route to specialized orchestrators
Maintainability	Clear responsibility ownership
8. Benefits

Predictable debugging

Enterprise maintainability

Clean logging

Consistent request tracing

Easy GPU service integration

Easy model swapping

Safe concurrent processing

Production-grade reliability

9. What This Architecture Is NOT

It is NOT:

MVC

Monolithic service class

Fat route system

ORM-as-business-logic pattern

Repository-only abstraction

Microservice sprawl

It is a layered, transaction-safe domain architecture.

10. Future Expansion Support

This structure cleanly supports:

Automatic chunk splitting

Automatic embedding generation

Image extraction from markdown

Unified ingestion response model

Dataset governance

Model version control

MLflow tracking

Multi-tenant site isolation

Air-gapped deployments

11. Summary

The EMTAC architecture is built on:

Adapters → Coordinators → Orchestrators → Services → Database

Where:

Adapters shape external input

Coordinators manage workflow

Orchestrators manage transactions

Services manage domain logic

Database layer manages persistence

This separation ensures reliability, scalability, and production-level system stability.