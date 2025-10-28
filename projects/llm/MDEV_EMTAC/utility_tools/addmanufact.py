# Script to add all manufacturers as keywords to FIND_BY_MANUFACTURER intent

def add_manufacturers_as_keywords(session):
    """
    Query the database for all manufacturers and add them as keywords
    to the FIND_BY_MANUFACTURER intent (ID 19).
    """
    try:
        from modules.emtacdb.emtacdb_fts import Part
        from modules.search.models.search_models import IntentKeyword

        # Get all unique manufacturers from the parts table
        manufacturers = session.query(Part.oem_mfg).distinct().filter(
            Part.oem_mfg.isnot(None),
            Part.oem_mfg != '',
            Part.oem_mfg != 'Unknown'
        ).all()

        # Extract manufacturer names
        manufacturer_names = [mfg[0].strip() for mfg in manufacturers if mfg[0] and mfg[0].strip()]

        print(f"Found {len(manufacturer_names)} unique manufacturers:")
        for mfg in sorted(manufacturer_names):
            print(f"  - {mfg}")

        # Add each manufacturer as a keyword to FIND_BY_MANUFACTURER intent (ID 19)
        intent_id = 19  # FIND_BY_MANUFACTURER
        added_count = 0
        updated_count = 0

        for manufacturer in manufacturer_names:
            # Check if keyword already exists
            existing = session.query(IntentKeyword).filter_by(
                intent_id=intent_id,
                keyword_text=manufacturer
            ).first()

            if existing:
                # Update existing keyword
                existing.weight = 1.5  # High weight for manufacturers
                existing.is_exact_match = False  # Allow partial matches
                existing.is_active = True
                updated_count += 1
                print(f"  ✓ Updated: {manufacturer}")
            else:
                # Add new keyword
                new_keyword = IntentKeyword(
                    intent_id=intent_id,
                    keyword_text=manufacturer,
                    weight=1.5,  # High weight for manufacturers
                    is_exact_match=False,  # Allow partial matches like "banner" matching "Banner"
                    is_active=True
                )
                session.add(new_keyword)
                added_count += 1
                print(f"  + Added: {manufacturer}")

        # Commit changes
        session.commit()

        print(f"\n✅ Summary:")
        print(f"   Added: {added_count} new manufacturer keywords")
        print(f"   Updated: {updated_count} existing keywords")
        print(f"   Total manufacturers: {len(manufacturer_names)}")

        return {
            'status': 'success',
            'added': added_count,
            'updated': updated_count,
            'total': len(manufacturer_names),
            'manufacturers': manufacturer_names
        }

    except Exception as e:
        session.rollback()
        print(f"❌ Error: {e}")
        return {'status': 'error', 'message': str(e)}


# Script to run this
if __name__ == "__main__":
    from modules.configuration.config_env import DatabaseConfig

    # Get database session
    db_config = DatabaseConfig()
    session = db_config.get_main_session()

    try:
        # Add manufacturers as keywords
        result = add_manufacturers_as_keywords(session)

        if result['status'] == 'success':
            print(f"\n🎉 Successfully updated FIND_BY_MANUFACTURER intent!")
            print(f"Now 'Banner sensors' should trigger the manufacturer search with higher confidence.")

            # Show some example manufacturers
            print(f"\nExample manufacturers that will now trigger FIND_BY_MANUFACTURER:")
            for mfg in sorted(result['manufacturers'])[:10]:
                print(f"  - '{mfg} sensors' will now match FIND_BY_MANUFACTURER")

    finally:
        session.close()

# Alternative: SQL script to do the same thing
sql_script = """
-- SQL script to add all manufacturers as keywords to FIND_BY_MANUFACTURER intent

-- First, let's see what manufacturers we have
SELECT DISTINCT oem_mfg, COUNT(*) as part_count 
FROM part 
WHERE oem_mfg IS NOT NULL 
  AND oem_mfg != '' 
  AND oem_mfg != 'Unknown'
GROUP BY oem_mfg 
ORDER BY part_count DESC;

-- Insert manufacturers as keywords (replace duplicates)
INSERT INTO intent_keyword (intent_id, keyword_text, weight, is_exact_match, is_active, created_at, updated_at)
SELECT DISTINCT 
    19 as intent_id,  -- FIND_BY_MANUFACTURER intent ID
    oem_mfg as keyword_text,
    1.5 as weight,     -- High weight for manufacturers
    false as is_exact_match,  -- Allow partial matches
    true as is_active,
    NOW() as created_at,
    NOW() as updated_at
FROM part 
WHERE oem_mfg IS NOT NULL 
  AND oem_mfg != '' 
  AND oem_mfg != 'Unknown'
  AND oem_mfg NOT IN (
    -- Don't add if already exists
    SELECT keyword_text 
    FROM intent_keyword 
    WHERE intent_id = 19
  );

-- Verify what was added
SELECT ik.keyword_text, ik.weight, ik.is_exact_match
FROM intent_keyword ik
WHERE ik.intent_id = 19  -- FIND_BY_MANUFACTURER
ORDER BY ik.keyword_text;
"""

print("=" * 60)
print("SOLUTION: Add Manufacturers as Keywords")
print("=" * 60)
print("1. Run the Python script above to automatically add all manufacturers")
print("2. OR run the SQL script to do it manually")
print("3. After adding, 'Banner sensors' should trigger FIND_BY_MANUFACTURER instead of FIND_SENSOR")
print("4. The search will then use oem_mfg='Banner' + equipment filter")