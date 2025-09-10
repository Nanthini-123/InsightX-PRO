# modules/emailer.py
import smtplib
from email.message import EmailMessage
import os

def send_email_with_attachment(smtp_host, smtp_port, username, password, to_email, subject, body, attachment_path):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = username
    msg['To'] = to_email
    msg.set_content(body)
    with open(attachment_path, 'rb') as f:
        data = f.read()
    msg.add_attachment(data, maintype='application', subtype='pdf', filename=os.path.basename(attachment_path))
    with smtplib.SMTP_SSL(smtp_host, smtp_port) as smtp:
        smtp.login(username, password)
        smtp.send_message(msg)
    return True