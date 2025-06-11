import gspread
from oauth2client.service_account import ServiceAccountCredentials

def update_dashboard(report):
    scope = ["https://spreadsheets.google.com/feeds",  "https://www.googleapis.com/auth/drive"] 
    creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open("SAAStelligence Dashboard").sheet1
    sheet.update_row(2, list(report.values()))