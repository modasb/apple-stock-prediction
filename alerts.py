import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import ssl  # Add SSL for secure connection
import traceback
import json
import os

class AlertSystem:
    def __init__(self):
        # Email configuration (Using Gmail)
        self.email_sender = "oussemadakhli1945@gmail.com"
        self.email_password = "oiic spht cygs mseq"
        self.email_receiver = "mariem.ayadi001@gmail.com"
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        
        # Use absolute path for receivers file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.receivers_file = os.path.join(base_dir, "email_receivers.json")
        print(f"Receivers file path: {self.receivers_file}")
        
        self.receivers = self.load_receivers()
        
    def load_receivers(self):
        try:
            # Get absolute path
            abs_path = os.path.abspath(self.receivers_file)
            print(f"Looking for receivers file at: {abs_path}")
            
            if os.path.exists(abs_path):
                print("File exists, loading receivers...")
                with open(abs_path, 'r') as f:
                    receivers = json.load(f)
                    print(f"Loaded {len(receivers)} receivers: {receivers}")
                    return receivers
            else:
                print("File doesn't exist, creating new one...")
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                # Create empty file
                with open(abs_path, 'w') as f:
                    json.dump([], f)
                print("Created new empty receivers file")
                return []
        except Exception as e:
            print(f"Error loading receivers: {e}")
            print(traceback.format_exc())
            return []
    
    def save_receivers(self):
        try:
            abs_path = os.path.abspath(self.receivers_file)
            print(f"Saving receivers to: {abs_path}")
            print(f"Current receivers: {self.receivers}")
            
            with open(abs_path, 'w') as f:
                json.dump(self.receivers, f)
            print("Receivers saved successfully")
            return True
        except Exception as e:
            print(f"Error saving receivers: {e}")
            print(traceback.format_exc())
            return False

    def add_receiver(self, email):
        print(f"\nAttempting to add email: {email}")
        print(f"Current receivers: {self.receivers}")
        
        if not email:
            print("Email is empty")
            return False
        
        if email not in self.receivers:
            print("Email not in list, adding...")
            self.receivers.append(email)
            if self.save_receivers():
                print(f"Email {email} added successfully")
                return True
            else:
                print("Failed to save receivers")
                return False
        else:
            print(f"Email {email} already exists")
            return False

    def remove_receiver(self, email):
        if email in self.receivers:
            self.receivers.remove(email)
            self.save_receivers()
            return True
        return False

    def send_test_email(self):
        """Function to send a test email with detailed debugging"""
        try:
            print("\nStarting email test...")
            
            # Create message
            print("Creating email message...")
            msg = MIMEMultipart()
            msg['From'] = self.email_sender
            msg['To'] = self.email_receiver
            msg['Subject'] = "Test Email from Stock Alert System"

            body = f"""
            Test Email from Stock Prediction System!
            
            Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            If you received this email, the alert system is working correctly.
            """
            msg.attach(MIMEText(body, 'plain'))
            print("Email message created successfully")

            # Create secure SSL/TLS context
            context = ssl.create_default_context()
            
            # Connect to SMTP server
            print(f"\nConnecting to {self.smtp_server}:{self.smtp_port}...")
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.set_debuglevel(1)  # Enable SMTP debug output
                
                print("Starting TLS encryption...")
                server.starttls(context=context)
                
                print("Attempting login...")
                server.login(self.email_sender, self.email_password)
                print("Login successful!")
                
                print("Sending email...")
                server.send_message(msg)
                print("Email sent successfully!")

            return True
            
        except smtplib.SMTPAuthenticationError as e:
            print(f"\nAuthentication failed!")
            print(f"Error code: {e.smtp_code}")
            print(f"Error message: {e.smtp_error}")
            return False
            
        except smtplib.SMTPException as e:
            print(f"\nSMTP error occurred!")
            print(f"Error: {str(e)}")
            return False
            
        except Exception as e:
            print(f"\nUnexpected error occurred!")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {str(e)}")
            return False

    def check_price_movement(self, current_price, previous_price):
        if not previous_price:
            print("No previous price available for comparison")
            return
        
        change_percent = ((current_price - previous_price) / previous_price) * 100
        print(f"Price change: {change_percent:.2f}%")
        
        if change_percent <= -1:
            print(f"Significant price drop detected: {change_percent:.2f}%")
            self.send_email_alert(current_price, change_percent, "PRICE DROP")
        elif change_percent >= 1:
            print(f"Significant price rise detected: {change_percent:.2f}%")
            self.send_email_alert(current_price, change_percent, "PRICE RISE")
    
    def send_email_alert(self, price, change, alert_type):
        try:
            print(f"\nAttempting to send {alert_type} alert to {len(self.receivers)} receivers...")
            
            for receiver in self.receivers:
                msg = MIMEMultipart()
                msg['From'] = self.email_sender
                msg['To'] = receiver
                msg['Subject'] = f"AAPL Stock {alert_type} Alert!"

                body = f"""
                AAPL Stock Alert!
                
                Current Price: ${price:.2f}
                Change: {change:.2f}%
                Type: {alert_type}
                Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                This is an automated alert from your Stock Prediction System.
                """
                
                msg.attach(MIMEText(body, 'plain'))
                
                with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                    server.starttls()
                    server.login(self.email_sender, self.email_password)
                    server.send_message(msg)
                    print(f"Alert sent to {receiver}")
                
        except Exception as e:
            print(f"\nError sending alert email:")
            print(traceback.format_exc())

# Test the email system
if __name__ == "__main__":
    alert = AlertSystem()
    print("Starting email system test...")
    result = alert.send_test_email()
    if result:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed!")