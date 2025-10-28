import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# import your Base and Position model
from modules.emtacdb.emtacdb_fts import Base, Position


@pytest.fixture
def session():
    # In-memory SQLite for isolated tests
    engine = create_engine('sqlite:///:memory:')
    # Create all tables
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def test_add_to_db_prevents_duplicates(session):
    """
    Calling add_to_db twice with identical FK args should return the same Position
    instance (no duplicate row).
    """
    # first insertion
    pos1 = Position.add_to_db(
        session,
        area_id=1,
        equipment_group_id=2,
        model_id=3,
        asset_number_id=4,
        location_id=5,
        subassembly_id=None,
        component_assembly_id=None,
        assembly_view_id=None,
        site_location_id=6
    )
    assert pos1.id == 1

    # second call with the exact same IDs
    pos2 = Position.add_to_db(
        session,
        area_id=1,
        equipment_group_id=2,
        model_id=3,
        asset_number_id=4,
        location_id=5,
        subassembly_id=None,
        component_assembly_id=None,
        assembly_view_id=None,
        site_location_id=6
    )
    # should get the same instance back
    assert pos2.id == pos1.id

    # only one row in the table
    assert session.query(Position).count() == 1


def test_add_to_db_creates_new_on_different_args(session):
    """
    If any FK differs, add_to_db should create a new row.
    """
    # insert first row
    pos1 = Position.add_to_db(
        session,
        area_id=10,
        equipment_group_id=None,
        model_id=None,
        asset_number_id=None,
        location_id=None,
        subassembly_id=None,
        component_assembly_id=None,
        assembly_view_id=None,
        site_location_id=None
    )
    assert pos1.id == 1

    # insert second row with a different area_id
    pos2 = Position.add_to_db(
        session,
        area_id=11,
        equipment_group_id=None,
        model_id=None,
        asset_number_id=None,
        location_id=None,
        subassembly_id=None,
        component_assembly_id=None,
        assembly_view_id=None,
        site_location_id=None
    )
    assert pos2.id == 2
    assert pos2.id != pos1.id

    # calling again with area_id=11 returns the same second row
    pos3 = Position.add_to_db(
        session,
        area_id=11,
        equipment_group_id=None,
        model_id=None,
        asset_number_id=None,
        location_id=None,
        subassembly_id=None,
        component_assembly_id=None,
        assembly_view_id=None,
        site_location_id=None
    )
    assert pos3.id == pos2.id

    # exactly two rows total
    assert session.query(Position).count() == 2
