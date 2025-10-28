// static/js/tool/position_dependencies.js

document.addEventListener('DOMContentLoaded', function() {
    // Get all the position form select elements
    const areaSelect = document.querySelector('select[name="area"]');
    const equipmentGroupSelect = document.querySelector('select[name="equipment_group"]');
    const modelSelect = document.querySelector('select[name="model"]');
    const assetNumberSelect = document.querySelector('select[name="asset_number"]');
    const locationSelect = document.querySelector('select[name="location"]');
    const subassemblySelect = document.querySelector('select[name="subassembly"]');
    const componentAssemblySelect = document.querySelector('select[name="component_assembly"]');
    const assemblyViewSelect = document.querySelector('select[name="assembly_view"]');
    const siteLocationSelect = document.querySelector('select[name="site_location"]');

    // Skip if we're not on a page with position form elements
    if (!areaSelect) return;

    console.log('Position dependencies initialized');

    // Function to reset a select element options (except the first one)
    function resetSelect(select) {
        if (!select) return;

        // Keep only the blank/default option
        const defaultOption = select.querySelector('option[value="__None"]') ||
                              select.querySelector('option[value=""]');

        if (defaultOption) {
            select.innerHTML = '';
            select.appendChild(defaultOption.cloneNode(true));

            // If using Select2, update it
            $(select).val(defaultOption.value).trigger('change');
        }
    }

    // Function to fetch dependent items
async function fetchDependentItems(parentType, parentId, childType = null) {
    try {
        // Add the '/tool' prefix to the URL
        let url = `/tool/get_dependent_items?parent_type=${parentType}&parent_id=${parentId}`;
        if (childType) {
            url += `&child_type=${childType}`;
        }

        console.log(`Fetching from: ${url}`);
        const response = await fetch(url);

        if (!response.ok) {
            throw new Error(`Failed to fetch options: ${response.statusText}`);
        }

        return await response.json();
    } catch (error) {
        console.error(`Error fetching dependent items for ${parentType}:`, error);
        return [];
    }
}

    // Function to update a select element with new options
    function updateSelectOptions(select, options, labelProperty = 'name') {
    if (!select) return;

    // Keep the default option
    const defaultOption = select.querySelector('option[value="__None"]') ||
                          select.querySelector('option[value=""]');

    select.innerHTML = '';
    if (defaultOption) {
        select.appendChild(defaultOption.cloneNode(true));
    }

    // Add new options
    options.forEach(item => {
        const option = document.createElement('option');
        option.value = item.id;

        // Handle different label formats
        if (select === assetNumberSelect && item.number) {
            // For asset numbers, use the number field
            option.textContent = item.number;
        } else if (labelProperty === 'site_location') {
            option.textContent = `${item.title} - Room ${item.room_number}`;
        } else {
            option.textContent = item[labelProperty] || item.id;
        }

        select.appendChild(option);
    });

    // If using Select2, update it
    $(select).trigger('change');
}

    // Area change handler
    if (areaSelect) {
        areaSelect.addEventListener('change', async function() {
            // Clear all dependent dropdowns first
            resetSelect(equipmentGroupSelect);
            resetSelect(modelSelect);
            resetSelect(assetNumberSelect);
            resetSelect(locationSelect);
            resetSelect(subassemblySelect);
            resetSelect(componentAssemblySelect);
            resetSelect(assemblyViewSelect);

            const areaId = this.value;
            if (areaId && areaId !== '__None') {
                console.log('Area changed to:', areaId);

                try {
                    const equipmentGroups = await fetchDependentItems('area', areaId);
                    updateSelectOptions(equipmentGroupSelect, equipmentGroups);
                } catch (error) {
                    console.error('Failed to update equipment groups:', error);
                }
            }
        });
    }

    // Equipment Group change handler
    if (equipmentGroupSelect) {
        equipmentGroupSelect.addEventListener('change', async function() {
            // Clear dependent dropdowns
            resetSelect(modelSelect);
            resetSelect(assetNumberSelect);
            resetSelect(locationSelect);
            resetSelect(subassemblySelect);
            resetSelect(componentAssemblySelect);
            resetSelect(assemblyViewSelect);

            const equipmentGroupId = this.value;
            if (equipmentGroupId && equipmentGroupId !== '__None') {
                console.log('Equipment Group changed to:', equipmentGroupId);

                try {
                    const models = await fetchDependentItems('equipment_group', equipmentGroupId);
                    updateSelectOptions(modelSelect, models);
                } catch (error) {
                    console.error('Failed to update models:', error);
                }
            }
        });
    }

    // Model change handler
    if (modelSelect) {
        modelSelect.addEventListener('change', async function() {
            // Clear dependent dropdowns
            resetSelect(assetNumberSelect);
            resetSelect(locationSelect);
            resetSelect(subassemblySelect);
            resetSelect(componentAssemblySelect);
            resetSelect(assemblyViewSelect);

            const modelId = this.value;
            if (modelId && modelId !== '__None') {
                console.log('Model changed to:', modelId);

                try {
                    // Update both asset numbers and locations using specific child types
                    const assetNumbers = await fetchDependentItems('model', modelId, 'asset_number');
                    updateSelectOptions(assetNumberSelect, assetNumbers, 'number');

                    const locations = await fetchDependentItems('model', modelId, 'location');
                    updateSelectOptions(locationSelect, locations);
                } catch (error) {
                    console.error('Failed to update asset numbers or locations:', error);
                }
            }
        });
    }

    // Location change handler
    if (locationSelect) {
        locationSelect.addEventListener('change', async function() {
            // Clear dependent dropdowns
            resetSelect(subassemblySelect);
            resetSelect(componentAssemblySelect);
            resetSelect(assemblyViewSelect);

            const locationId = this.value;
            if (locationId && locationId !== '__None') {
                console.log('Location changed to:', locationId);

                try {
                    const subassemblies = await fetchDependentItems('location', locationId);
                    updateSelectOptions(subassemblySelect, subassemblies);
                } catch (error) {
                    console.error('Failed to update subassemblies:', error);
                }
            }
        });
    }

    // Subassembly change handler
    if (subassemblySelect) {
        subassemblySelect.addEventListener('change', async function() {
            // Clear dependent dropdowns
            resetSelect(componentAssemblySelect);
            resetSelect(assemblyViewSelect);

            const subassemblyId = this.value;
            if (subassemblyId && subassemblyId !== '__None') {
                console.log('Subassembly changed to:', subassemblyId);

                try {
                    const componentAssemblies = await fetchDependentItems('subassembly', subassemblyId);
                    updateSelectOptions(componentAssemblySelect, componentAssemblies);
                } catch (error) {
                    console.error('Failed to update component assemblies:', error);
                }
            }
        });
    }

    // Component Assembly change handler
    if (componentAssemblySelect) {
        componentAssemblySelect.addEventListener('change', async function() {
            // Clear dependent dropdowns
            resetSelect(assemblyViewSelect);

            const componentAssemblyId = this.value;
            if (componentAssemblyId && componentAssemblyId !== '__None') {
                console.log('Component Assembly changed to:', componentAssemblyId);

                try {
                    const assemblyViews = await fetchDependentItems('component_assembly', componentAssemblyId);
                    updateSelectOptions(assemblyViewSelect, assemblyViews);
                } catch (error) {
                    console.error('Failed to update assembly views:', error);
                }
            }
        });
    }
});