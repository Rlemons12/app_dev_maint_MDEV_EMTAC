document.addEventListener('DOMContentLoaded', function () {
    // ============================================================
    //  Grab all the selects we care about
    // ============================================================
    const campusSelect            = document.getElementById('campus');
    const buildingSelect          = document.getElementById('building');
    const siteLocationSelect      = document.getElementById('site_location');
    const areaSelect              = document.getElementById('area');
    const equipmentGroupSelect    = document.getElementById('equipment_group');
    const modelSelect             = document.getElementById('model');
    const assetNumberSelect       = document.getElementById('asset_number');
    const locationSelect          = document.getElementById('location');
    const subassemblySelect       = document.getElementById('subassembly');
    const componentAssemblySelect = document.getElementById('component_assembly');
    const assemblyViewSelect      = document.getElementById('assembly_view');

    // Helper to reset a select to a single placeholder option and disable it
    function resetSelect(selectEl, placeholderText) {
        if (!selectEl) return;
        selectEl.innerHTML = `<option value="">${placeholderText}</option>`;
        selectEl.disabled = true;
    }

    // Helper to enable a select
    function enableSelect(selectEl) {
        if (!selectEl) return;
        selectEl.disabled = false;
    }

    // ============================================================
    //  Initial reset state (optional but nice)
    // ============================================================
    if (buildingSelect)          resetSelect(buildingSelect,          'Select Building');
    if (siteLocationSelect)      resetSelect(siteLocationSelect,      'Select Site Location');
    if (areaSelect)              resetSelect(areaSelect,              'Select Area');
    if (equipmentGroupSelect)    resetSelect(equipmentGroupSelect,    'Select Equipment Group');
    if (modelSelect)             resetSelect(modelSelect,             'Select Model');
    if (assetNumberSelect)       resetSelect(assetNumberSelect,       'Select Asset Number');
    if (locationSelect)          resetSelect(locationSelect,          'Select Location');
    if (subassemblySelect)       resetSelect(subassemblySelect,       'Select Subassembly');
    if (componentAssemblySelect) resetSelect(componentAssemblySelect, 'Select Component Assembly');
    if (assemblyViewSelect)      resetSelect(assemblyViewSelect,      'Select Assembly View');

    // ============================================================
    //  CAMPUS → BUILDING dynamic filter
    // ============================================================
    if (campusSelect && buildingSelect) {
        campusSelect.addEventListener('change', function () {
            const campusId = this.value;

            // Reset downstream
            resetSelect(buildingSelect,          'Select Building');
            if (siteLocationSelect) resetSelect(siteLocationSelect, 'Select Site Location');
            if (areaSelect)         resetSelect(areaSelect,         'Select Area');

            resetSelect(equipmentGroupSelect,    'Select Equipment Group');
            resetSelect(modelSelect,             'Select Model');
            resetSelect(assetNumberSelect,       'Select Asset Number');
            resetSelect(locationSelect,          'Select Location');
            resetSelect(subassemblySelect,       'Select Subassembly');
            resetSelect(componentAssemblySelect, 'Select Component Assembly');
            resetSelect(assemblyViewSelect,      'Select Assembly View');

            if (!campusId) return; // nothing selected

            fetch(`/get_buildings?campus_id=${campusId}`)
                .then(response => response.json())
                .then(data => {
                    data.forEach(building => {
                        const option = document.createElement('option');
                        option.value = building.id;
                        option.textContent = building.name;
                        buildingSelect.appendChild(option);
                    });
                    enableSelect(buildingSelect);
                })
                .catch(error => {
                    console.error("Error loading buildings:", error);
                    alert("Failed to load buildings for the selected campus.");
                });
        });
    } else {
        console.warn("Campus or Building dropdown not found — dynamic filtering not applied.");
    }

    // ============================================================
    //  BUILDING → SITE LOCATION dynamic filter
    //  (Site Location is ONLY tied to Building, not Area)
    // ============================================================
    if (buildingSelect && siteLocationSelect) {
        buildingSelect.addEventListener('change', function () {
            const buildingId = this.value;

            // Reset site locations
            resetSelect(siteLocationSelect, 'Select Site Location');

            // Reset Area + downstream chain
            if (areaSelect)         resetSelect(areaSelect,         'Select Area');
            resetSelect(equipmentGroupSelect,    'Select Equipment Group');
            resetSelect(modelSelect,             'Select Model');
            resetSelect(assetNumberSelect,       'Select Asset Number');
            resetSelect(locationSelect,          'Select Location');
            resetSelect(subassemblySelect,       'Select Subassembly');
            resetSelect(componentAssemblySelect, 'Select Component Assembly');
            resetSelect(assemblyViewSelect,      'Select Assembly View');

            // Optionally: only allow area selection after a building is chosen
            if (areaSelect) areaSelect.disabled = true;

            if (!buildingId) return;

            fetch(`/get_site_locations_by_building?building_id=${buildingId}`)
                .then(response => response.json())
                .then(data => {
                    data.forEach(loc => {
                        const option = document.createElement('option');
                        option.value = loc.id;
                        option.textContent = `${loc.title} (Room ${loc.room_number})`;
                        siteLocationSelect.appendChild(option);
                    });
                    enableSelect(siteLocationSelect);
                })
                .catch(error => {
                    console.error("Error loading site locations:", error);
                    alert("Failed to load site locations for this building.");
                });
        });
    }

    // ============================================================
    //  SITE LOCATION → AREA dynamic filter
    //  Uses /get_areas_by_site_location and Position as the bridge
    // ============================================================
    if (siteLocationSelect && areaSelect) {
        siteLocationSelect.addEventListener('change', function () {
            const siteLocationId = this.value;

            // Reset Area + everything that depends on Area
            resetSelect(areaSelect,              'Select Area');
            resetSelect(equipmentGroupSelect,    'Select Equipment Group');
            resetSelect(modelSelect,             'Select Model');
            resetSelect(assetNumberSelect,       'Select Asset Number');
            resetSelect(locationSelect,          'Select Location');
            resetSelect(subassemblySelect,       'Select Subassembly');
            resetSelect(componentAssemblySelect, 'Select Component Assembly');
            resetSelect(assemblyViewSelect,      'Select Assembly View');

            if (!siteLocationId) {
                // No site location selected → leave Area disabled.
                return;
            }

            fetch(`/get_areas_by_site_location?site_location_id=${siteLocationId}`)
                .then(response => response.json())
                .then(data => {
                    if (!data || data.length === 0) {
                        // No existing Areas for this site location.
                        // User can still type a new Area in the area_input field.
                        return;
                    }

                    data.forEach(area => {
                        const option = document.createElement('option');
                        option.value = area.id;
                        option.textContent = area.name;
                        areaSelect.appendChild(option);
                    });

                    enableSelect(areaSelect);
                })
                .catch(err => {
                    console.error('Error fetching areas for site location:', err);
                    alert('An error occurred while fetching areas for this site location.');
                });
        });
    }

    // ============================================================
    //  AREA → EQUIPMENT GROUP (via /get_equipment_groups)
    // ============================================================
    if (areaSelect && equipmentGroupSelect) {
        areaSelect.addEventListener('change', function () {
            const areaId = this.value;

            resetSelect(equipmentGroupSelect,    'Select Equipment Group');
            resetSelect(modelSelect,             'Select Model');
            resetSelect(assetNumberSelect,       'Select Asset Number');
            resetSelect(locationSelect,          'Select Location');
            resetSelect(subassemblySelect,       'Select Subassembly');
            resetSelect(componentAssemblySelect, 'Select Component Assembly');
            resetSelect(assemblyViewSelect,      'Select Assembly View');

            if (!areaId) return;

            fetch(`/get_equipment_groups?area_id=${areaId}`)
                .then(response => response.json())
                .then(data => {
                    data.forEach(group => {
                        const option = document.createElement('option');
                        option.value = group.id;
                        option.textContent = group.name;
                        equipmentGroupSelect.appendChild(option);
                    });
                    enableSelect(equipmentGroupSelect);
                })
                .catch(err => {
                    console.error('Error fetching equipment groups:', err);
                    alert('An error occurred while fetching equipment groups.');
                });
        });
    }

    // ============================================================
    //  EQUIPMENT GROUP → MODEL (/get_models)
    // ============================================================
    if (equipmentGroupSelect && modelSelect) {
        equipmentGroupSelect.addEventListener('change', function () {
            const egId = this.value;

            resetSelect(modelSelect,             'Select Model');
            resetSelect(assetNumberSelect,       'Select Asset Number');
            resetSelect(locationSelect,          'Select Location');
            resetSelect(subassemblySelect,       'Select Subassembly');
            resetSelect(componentAssemblySelect, 'Select Component Assembly');
            resetSelect(assemblyViewSelect,      'Select Assembly View');

            if (!egId) return;

            fetch(`/get_models?equipment_group_id=${egId}`)
                .then(response => response.json())
                .then(data => {
                    data.forEach(model => {
                        const option = document.createElement('option');
                        option.value = model.id;
                        option.textContent = model.name;
                        modelSelect.appendChild(option);
                    });
                    enableSelect(modelSelect);
                })
                .catch(err => {
                    console.error('Error fetching models:', err);
                    alert('An error occurred while fetching models.');
                });
        });
    }

    // ============================================================
    //  MODEL → ASSET NUMBER & LOCATION
    // ============================================================
    if (modelSelect) {
        modelSelect.addEventListener('change', function () {
            const modelId = this.value;

            resetSelect(assetNumberSelect,       'Select Asset Number');
            resetSelect(locationSelect,          'Select Location');
            resetSelect(subassemblySelect,       'Select Subassembly');
            resetSelect(componentAssemblySelect, 'Select Component Assembly');
            resetSelect(assemblyViewSelect,      'Select Assembly View');

            if (!modelId) return;

            // Asset numbers
            if (assetNumberSelect) {
                fetch(`/get_asset_numbers?model_id=${modelId}`)
                    .then(response => response.json())
                    .then(data => {
                        data.forEach(asset => {
                            const option = document.createElement('option');
                            option.value = asset.id;
                            option.textContent = asset.number;
                            assetNumberSelect.appendChild(option);
                        });
                        enableSelect(assetNumberSelect);
                    })
                    .catch(err => {
                        console.error('Error fetching asset numbers:', err);
                        alert('An error occurred while fetching asset numbers.');
                    });
            }

            // Locations
            if (locationSelect) {
                fetch(`/get_locations?model_id=${modelId}`)
                    .then(response => response.json())
                    .then(data => {
                        data.forEach(loc => {
                            const option = document.createElement('option');
                            option.value = loc.id;
                            option.textContent = loc.name;
                            locationSelect.appendChild(option);
                        });
                        enableSelect(locationSelect);
                    })
                    .catch(err => {
                        console.error('Error fetching locations:', err);
                        alert('An error occurred while fetching locations.');
                    });
            }
        });
    }

    // ============================================================
    //  LOCATION → SUBASSEMBLY
    // ============================================================
    if (locationSelect && subassemblySelect) {
        locationSelect.addEventListener('change', function () {
            const locationId = this.value;

            resetSelect(subassemblySelect,       'Select Subassembly');
            resetSelect(componentAssemblySelect, 'Select Component Assembly');
            resetSelect(assemblyViewSelect,      'Select Assembly View');

            if (!locationId) return;

            fetch(`/get_subassemblies?location_id=${locationId}`)
                .then(response => response.json())
                .then(data => {
                    data.forEach(sub => {
                        const option = document.createElement('option');
                        option.value = sub.id;
                        option.textContent = sub.name;
                        subassemblySelect.appendChild(option);
                    });
                    enableSelect(subassemblySelect);
                })
                .catch(err => {
                    console.error('Error fetching subassemblies:', err);
                    alert('An error occurred while fetching subassemblies.');
                });
        });
    }

    // ============================================================
    //  SUBASSEMBLY → COMPONENT ASSEMBLY
    // ============================================================
    if (subassemblySelect && componentAssemblySelect) {
        subassemblySelect.addEventListener('change', function () {
            const subassemblyId = this.value;

            resetSelect(componentAssemblySelect, 'Select Component Assembly');
            resetSelect(assemblyViewSelect,      'Select Assembly View');

            if (!subassemblyId) return;

            fetch(`/component_assemblies?subassembly_id=${subassemblyId}`)
                .then(response => response.json())
                .then(data => {
                    data.forEach(comp => {
                        const option = document.createElement('option');
                        option.value = comp.id;
                        option.textContent = comp.name;
                        componentAssemblySelect.appendChild(option);
                    });
                    enableSelect(componentAssemblySelect);
                })
                .catch(err => {
                    console.error('Error fetching component assemblies:', err);
                    alert('An error occurred while fetching component assemblies.');
                });
        });
    }

    // ============================================================
    //  COMPONENT ASSEMBLY → ASSEMBLY VIEW
    // ============================================================
    if (componentAssemblySelect && assemblyViewSelect) {
        componentAssemblySelect.addEventListener('change', function () {
            const compId = this.value;

            resetSelect(assemblyViewSelect, 'Select Assembly View');

            if (!compId) return;

            fetch(`/get_assembly_views?component_assembly_id=${compId}`)
                .then(response => response.json())
                .then(data => {
                    data.forEach(view => {
                        const option = document.createElement('option');
                        option.value = view.id;
                        option.textContent = view.name;
                        assemblyViewSelect.appendChild(option);
                    });
                    enableSelect(assemblyViewSelect);
                })
                .catch(err => {
                    console.error('Error fetching assembly views:', err);
                    alert('An error occurred while fetching assembly views.');
                });
        });
    }

    // ============================================================
    //  Toggle between dropdown and “new entry”
    // ============================================================
    function toggleNewEntry(fieldName) {
        const existingDiv = document.getElementById(fieldName + "_existing");
        const newDiv      = document.getElementById(fieldName + "_new");

        if (!existingDiv || !newDiv) {
            console.warn(`Elements for field ${fieldName} not found.`);
            return;
        }

        if (existingDiv.style.display === "none" || newDiv.style.display === "block") {
            existingDiv.style.display = "block";
            newDiv.style.display      = "none";

            const inputs = newDiv.querySelectorAll("input, textarea");
            inputs.forEach(input => input.value = "");
        } else {
            existingDiv.style.display = "none";
            newDiv.style.display      = "block";
        }
    }

    // ============================================================
    //  Show loading spinner on submit
    // ============================================================
    const form = document.getElementById('createPositionForm');
    if (form) {
        form.addEventListener('submit', function () {
            const spinner = document.getElementById('loadingSpinner');
            if (spinner) spinner.style.display = "block";
        });
    } else {
        console.warn('Form with id "createPositionForm" not found.');
    }

    // Expose toggle function globally
    window.toggleNewEntry = toggleNewEntry;
});
