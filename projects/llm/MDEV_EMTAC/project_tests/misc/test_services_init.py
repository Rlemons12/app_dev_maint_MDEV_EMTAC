from modules.services import DBServices


def test_dbservices_imports_all_services():
    """Ensure DBServices initializes without raising errors."""
    services = DBServices()

    # Verify key services exist
    assert services.areas is not None
    assert services.equipment_groups is not None
    assert services.models is not None
    assert services.positions is not None
    assert services.parts is not None
    assert services.documents is not None
    assert services.images is not None

    # If we reached this point, initialization succeeded
    assert True
