# scripts/test_observability_trace.py

import uuid
import time
import threading
from datetime import datetime

from modules.configuration.config_env import get_db_config
from modules.observability.models import (
    TraceSession,
    TraceSpan,
)

db = get_db_config()


# ---------------------------------------------------------
# Helper: Create Trace Session
# ---------------------------------------------------------

def create_trace_session(session):
    trace = TraceSession(
        id=uuid.uuid4(),
        request_id="TEST-REQUEST-001",
        service_name="emtac_test_service",
        environment="development",
        started_at=datetime.utcnow(),
        status="running",
        sampled=True,
    )

    session.add(trace)
    session.flush()

    # Return primitive ID only (never pass ORM outside session)
    return trace.id


# ---------------------------------------------------------
# Span Creator
# ---------------------------------------------------------

def create_span(
    session,
    trace_id,
    name,
    depth,
    parent_id=None,
    sleep_time=0.1,
):

    span = TraceSpan(
        id=uuid.uuid4(),
        trace_id=trace_id,
        parent_span_id=parent_id,
        name=name,
        depth=depth,
        started_at=datetime.utcnow(),
        request_id="TEST-REQUEST-001",
        thread_id=threading.get_ident(),
        process_id=0,
        status="running",
    )

    session.add(span)
    session.flush()

    # simulate work
    start = time.time()
    time.sleep(sleep_time)
    duration = (time.time() - start) * 1000

    span.ended_at = datetime.utcnow()
    span.duration_ms = round(duration, 2)
    span.status = "success"

    return span.id, span.duration_ms


# ---------------------------------------------------------
# Main Test Execution
# ---------------------------------------------------------

def run_test():

    with db.main_session() as session:

        print("Creating trace session...")
        trace_id = create_trace_session(session)

        print("Creating root span...")
        root_id, root_duration = create_span(
            session,
            trace_id=trace_id,
            name="root_operation",
            depth=0,
            sleep_time=0.2,
        )

        print("Creating child span 1...")
        child1_id, child1_duration = create_span(
            session,
            trace_id=trace_id,
            name="child_operation_1",
            depth=1,
            parent_id=root_id,
            sleep_time=0.1,
        )

        print("Creating child span 2...")
        child2_id, child2_duration = create_span(
            session,
            trace_id=trace_id,
            name="child_operation_2",
            depth=1,
            parent_id=root_id,
            sleep_time=0.15,
        )

        # Update session summary safely
        trace = session.query(TraceSession).get(trace_id)
        trace.ended_at = datetime.utcnow()
        trace.duration_ms = round(
            root_duration + child1_duration + child2_duration, 2
        )
        trace.status = "success"

        print("Trace committed to database.")

    # -----------------------------------------------------
    # Verify
    # -----------------------------------------------------

    with db.main_session() as session:

        spans = (
            session.query(TraceSpan)
            .filter(TraceSpan.trace_id == trace_id)
            .order_by(TraceSpan.depth, TraceSpan.started_at)
            .all()
        )

        print("\nRetrieved Spans:")
        for s in spans:
            print(
                f"Span: {s.name} | depth={s.depth} | parent={s.parent_span_id} | duration={s.duration_ms}ms"
            )


if __name__ == "__main__":
    run_test()