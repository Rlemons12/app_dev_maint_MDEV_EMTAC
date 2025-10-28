import pyodbc

# Set up your connection parameters
server = 'ctac.database.windows.net'
database = 'ctac'
username = 'robert.lemons'
password = 'Texas1234'
driver = '{ODBC Driver 17 for SQL Server}'

# Establish the connection
conn = pyodbc.connect('DRIVER=' + driver + ';'
                      'SERVER=' + server + ';'
                      'DATABASE=' + database + ';'
                      'UID=' + username + ';'
                      'PWD=' + password)

cursor = conn.cursor()

# Execute a sample query
cursor.execute("SELECT * FROM your_table_name")
rows = cursor.fetchall()

for row in rows:
    print(row)

# Don't forget to close the connection
conn.close()
