from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def equipment_page():
    equipment = {
        "name": "High-Speed Bottle Filler",
        "equipment_id": "EQP-1021",
        "asset_number": "00024578",
        "manufacturer": "Krones AG",
        "model_number": "LVP-8000",
        "serial_number": "SN-4578923",
        "year_manufacture": 2021,
        "site_facility": "Main Plant - Line 2",
        "area": "Packaging Hall",
        "line_section": "Line 2",
        "location_code": "P2A-04",
        "image_url": "",
        "notes": "This filler was upgraded in 2023 to include automatic bottle size adjustment. Scheduled maintenance is every 3 months.",
    }
    related_docs = [
        {"name": "SOP-1001", "url": "#sop-1001"},
        {"name": "User Manual", "url": "#user-manual"},
        {"name": "Maintenance Log", "url": "#maintenance-log"},
    ]
    dropdown_sections = [
        {
            "title": "Quick Links",
            "items": [
                {"name": "Home", "url": "#home"},
                {"name": "Support", "url": "#support"},
            ]
        },
        {
            "title": "Equipment Actions",
            "items": [
                {"name": "Edit Equipment", "url": "#edit"},
                {"name": "View History", "url": "#history"},
            ]
        }
    ]
    # Use the correct relative path to your template:
    return render_template(
        "equipment_info/mock_equipment_info.html",
        equipment=equipment,
        related_docs=related_docs,
        dropdown_sections=dropdown_sections
    )

@app.route('/equipment/card-designer')
def equipment_card_designer():
    return render_template('equipment_card_design/eqcrddesigner.html')

if __name__ == "__main__":
    app.run(debug=True)
