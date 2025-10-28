import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

def send_email_with_attachment(sender_email, receiver_email, subject, message, attachment_path, password):
    try:
        # Create a multipart message and set headers
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject

        # Add message body
        msg.attach(MIMEText(message, 'plain'))

        # Open the file to be sent
        with open(attachment_path, "rb") as attachment:
            # Add file as application/octet-stream
            # Email client can usually download this automatically as attachment
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())

        # Encode file in ASCII characters to send by email    
        encoders.encode_base64(part)

        # Add header as key/value pair to attachment part
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {attachment_path}",
        )

        # Add attachment to message and convert message to string
        msg.attach(part)
        text = msg.as_string()

        # Log in to SMTP server and send email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, text)
        print("Email sent successfully!")
        return True
    except smtplib.SMTPException as e:
        print(f"SMTP error: {e}")
        return False
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

# Example usage
sender_email = "kermitsrocket@gmail.com"
receiver_email = "robert.lemons@icumed.com"
subject = "pyEmail"
message = "Please find the attached file. I think its a great speech!!!"
attachment_path = r"C:\Users\15127\OneDrive\Documents\ALexander the greate speach at Opis.docx"
password = "rycq reuv uycz hdlg"

if send_email_with_attachment(sender_email, receiver_email, subject, message, attachment_path, password):
    print("Email sent successfully!")
else:
    print("Failed to send email.")