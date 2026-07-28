# -*- coding: utf-8 -*-
import os
import base64
import json
from email.mime.text import MIMEText
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/drive.readonly'
]

CREDENTIALS_FILE = '/home/riza/ai-assistant/credentials.json'
TOKEN_FILE = '/home/riza/ai-assistant/token.json'

def get_google_service(service_name, version):
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    return build(service_name, version, credentials=creds)

def read_emails(max_results=5):
    try:
        service = get_google_service('gmail', 'v1')
        results = service.users().messages().list(
            userId='me', maxResults=max_results, q='is:unread'
        ).execute()
        messages = results.get('messages', [])
        if not messages:
            return "No unread emails."
        emails = []
        for msg in messages[:3]:
            msg_data = service.users().messages().get(
                userId='me', id=msg['id'], format='metadata',
                metadataHeaders=['From', 'Subject']
            ).execute()
            headers = {h['name']: h['value'] for h in msg_data['payload']['headers']}
            sender = headers.get('From', 'Unknown')
            subject = headers.get('Subject', 'No subject')
            emails.append(f"From {sender}: {subject}")
        return " | ".join(emails)
    except Exception as e:
        return f"Could not read emails: {e}"

def send_email(to, subject, body):
    try:
        service = get_google_service('gmail', 'v1')
        message = MIMEText(body)
        message['to'] = to
        message['subject'] = subject
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        service.users().messages().send(
            userId='me', body={'raw': raw}
        ).execute()
        return True
    except Exception as e:
        print(f"Send email error: {e}")
        return False

def get_calendar_events(max_results=5):
    try:
        from datetime import datetime, timezone
        service = get_google_service('calendar', 'v3')
        now = datetime.now(timezone.utc).isoformat()
        events_result = service.events().list(
            calendarId='primary', timeMin=now,
            maxResults=max_results, singleEvents=True,
            orderBy='startTime'
        ).execute()
        events = events_result.get('items', [])
        if not events:
            return "No upcoming events."
        result = []
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            result.append(f"{event['summary']} at {start}")
        return " | ".join(result)
    except Exception as e:
        return f"Could not get calendar: {e}"