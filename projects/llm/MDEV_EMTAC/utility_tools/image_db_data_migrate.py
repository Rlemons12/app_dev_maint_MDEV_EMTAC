import sqlite3

# Connect to the source database
source_conn = sqlite3.connect(r'C:\Users\10169062\Desktop\AI_EMTACr9\Database\1emtac_db.db')
source_cursor = source_conn.cursor()

# Connect to the destination database
dest_conn = sqlite3.connect(r'C:\Users\10169062\Desktop\AI_EMTACr9\Database\emtac_db.db')
dest_cursor = dest_conn.cursor()

# Extract data from the source table
source_cursor.execute('SELECT title, description, image_blob FROM images')
data = source_cursor.fetchall()

# Define the batch size
batch_size = 1000
total_images = len(data)

# Insert data into the destination table (image table) in batches
for i in range(0, total_images, batch_size):
    batch_data = data[i:i+batch_size]
    # Add values of 3 to the area column and 4 to the equipment_group column
    modified_batch_data = [(row[0], row[1], row[2], 3, 4,21) for row in batch_data]
    dest_cursor.executemany('INSERT INTO images (title, description, image_blob, area_id, equipment_group_id, model_id) VALUES (?,?, ?, ?, ?, ?)', modified_batch_data)
    dest_conn.commit()
    print(f"{min(i + batch_size, total_images)} images migrated.")
    print("Do you want to continue? (y/n)")
    user_input = input()
    if user_input.lower() != 'y':
        break

# Close the connections
source_conn.close()
dest_conn.close()

print("Migration completed successfully!")
