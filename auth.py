from google_auth_oauthlib.flow import InstalledAppFlow
import json

SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/drive.readonly'
]

flow = InstalledAppFlow.from_client_secrets_file(
    '/home/riza/ai-assistant/credentials.json',
    SCOPES,
    redirect_uri='urn:ietf:wg:oauth:2.0:oob'
)

auth_url, _ = flow.authorization_url(prompt='consent')
print('Open this URL in your browser:')
print(auth_url)
print()
code = input('Paste the code here: ')
flow.fetch_token(code=code)
creds = flow.credentials

with open('/home/riza/ai-assistant/token.json', 'w') as f:
    f.write(creds.to_json())

print('Done! token.json saved.')