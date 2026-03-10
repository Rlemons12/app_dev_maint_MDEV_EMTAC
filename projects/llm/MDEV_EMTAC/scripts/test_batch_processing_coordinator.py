from modules.coordinators.batch_processing_coordinator import BatchProcessingCoordinator


def main():
    folder_path = r"E:\emtac\data\test_batch_docs"

    metadata = {
        "title": "",
        "area": "1",
        "equipment_group": "2",
        "model": "3",
        "asset_number": "4",
        "location": "5",
        "site_location": "",
        "room_number": "Unknown",
        "department": "",
        "tags": "",
        "priority": "normal",
    }

    coordinator = BatchProcessingCoordinator()

    success, response, status = coordinator.process_folder(
        folder_path=folder_path,
        metadata=metadata,
        include_subfolders=True,
        concurrent=False,
        max_workers=4,
    )

    print("SUCCESS:", success)
    print("STATUS:", status)
    print("RESPONSE:")
    print(response)


if __name__ == "__main__":
    main()