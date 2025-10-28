import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Compose the email
msg = MIMEMultipart()
msg['From'] = 'your_email@example.com'
msg['To'] = 'recipient@example.com'
msg['Subject'] = 'Your Subject Here'

# Customize the email body with data from the database
body = ""
for row in data:
    body += f"Data: {row[0]}, {row[1]}, {row[2]}\n"  # Adjust according to your database structure

msg.attach(MIMEText(body, 'plain'))

# Connect to the SMTP server
server = smtplib.SMTP('smtp.example.com', 587)  # Use your SMTP server details
server.starttls()
server.login('your_email@example.com', 'your_password')

# Send the email
server.sendmail('your_email@example.com', 'recipient@example.com', msg.as_string())

# Close the connection
server.quit()
