"""
Integration Test: Tool Domain Wiring

Tests:
- Tool creation
- Image association
- Position association
- Aggregated retrieval
- Service alignment
"""

import os
import uuid

from modules.configuration.config_env import DatabaseConfig
from modules.emtac_ai.orchestrators.tool_orchestrator import ToolOrchestrator
from modules.emtacdb.emtacdb_fts import (
    Image,
    Position,
)


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

db_config = DatabaseConfig()
orchestrator = ToolOrchestrator(db_config)


def run_test():

    print("\n=== TOOL ORCHESTRATOR INTEGRATION TEST ===")

    tool_name = f"TEST_TOOL_{uuid.uuid4().hex[:8]}"

    # ------------------------------------------------------------
    # 1️⃣ Create Dummy Position + Image
    # ------------------------------------------------------------

    print("\n[1] Creating dummy Position + Image")

    with db_config.main_session() as session:

        position = Position()
        session.add(position)
        session.flush()

        image = Image(
            title="Test Image",
            description="Integration test image",
            file_path="/tmp/test_image.png",
        )
        session.add(image)
        session.flush()

        # Extract primitive IDs BEFORE commit
        position_id = position.id
        image_id = image.id

        session.commit()

    print(f"   Created Position ID: {position_id}")
    print(f"   Created Image ID: {image_id}")

    # ------------------------------------------------------------
    # 2️⃣ CREATE TOOL WITH RELATIONS
    # ------------------------------------------------------------

    print("\n[2] Creating Tool With Relations")

    tool_id = orchestrator.create_tool_with_relations(
        name=tool_name,
        size="1/2 inch",
        type_="Manual",
        material="Steel",
        description="Integration test tool",
        image_ids=[image_id],
        position_ids=[position_id],
    )

    assert tool_id is not None, "Tool creation failed"
    print(f"   Created Tool ID: {tool_id}")

    # ------------------------------------------------------------
    # 3️⃣ FETCH FULL PROFILE
    # ------------------------------------------------------------

    print("\n[3] Fetching Full Tool Profile")

    profile = orchestrator.get_full_tool_profile(tool_id)

    assert profile is not None, "Profile retrieval failed"
    assert profile["tool"]["name"] == tool_name
    assert len(profile["relations"]["images"]) == 1
    assert len(profile["relations"]["positions"]) == 1

    print("   Profile verified successfully")

    # ------------------------------------------------------------
    # 4️⃣ BULK ENRICHMENT TEST
    # ------------------------------------------------------------

    print("\n[4] Testing Enrichment By Position")

    enriched = orchestrator.enrich_tools_for_positions([position_id])

    assert len(enriched) >= 1
    assert any(t["id"] == tool_id for t in enriched)

    print("   Enrichment verified successfully")

    # ------------------------------------------------------------
    # 5️⃣ OPTIONAL CLEANUP (Recommended)
    # ------------------------------------------------------------

    print("\n[5] Cleaning up test data")

    with db_config.main_session() as session:

        from modules.emtacdb.emtacdb_fts import (
            Tool,
            ToolPositionAssociation,
            ToolImageAssociation,
        )

        # Remove associations first (FK safety)
        session.query(ToolPositionAssociation)\
            .filter_by(tool_id=tool_id)\
            .delete()

        session.query(ToolImageAssociation)\
            .filter_by(tool_id=tool_id)\
            .delete()

        session.query(Tool)\
            .filter_by(id=tool_id)\
            .delete()

        session.query(Image)\
            .filter_by(id=image_id)\
            .delete()

        session.query(Position)\
            .filter_by(id=position_id)\
            .delete()

        session.commit()

    print("   Cleanup complete")

    print("\n=== ALL TOOL TESTS PASSED ===\n")



if __name__ == "__main__":
    run_test()
