import pandas as pd
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.errors import HttpError
import os

# Authentication setup
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']


def authenticate_google_drive():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds


def list_images_in_folder(service, folder_id):
    try:
        image_files = []
        page_token = None

        while True:
            # Query to get the images from the folder
            query = f"'{folder_id}' in parents and mimeType contains 'image/'"
            results = service.files().list(
                q=query,
                fields="nextPageToken, files(id, name)",
                pageToken=page_token
            ).execute()

            # Append results
            image_files.extend(results.get('files', []))

            # Get the nextPageToken for pagination
            page_token = results.get('nextPageToken')

            # Break the loop if no more pages
            if page_token is None:
                break

        return image_files
    except HttpError as error:
        print(f"An error occurred: {error}")
        return None


def create_image_url(file_id):
    return f"https://drive.google.com/uc?id={file_id}"


def create_image_dataframe(folder_structure):
    creds = authenticate_google_drive()
    service = build('drive', 'v3', credentials=creds)

    image_data = []

    for label_type, folder_id in folder_structure.items():
        images = list_images_in_folder(service, folder_id)
        if images is None:
            print(f"No images found in {label_type}.")
            continue
        for image in images:
            file_url = create_image_url(image['id'])
            image_data.append([file_url, label_type])

    # Create DataFrame with URLs and labels
    df = pd.DataFrame(image_data, columns=["image_url", "label"])
    return df


if __name__ == '__main__':
    # Ensure that these folder IDs are correct and have proper sharing permissions
    folder_structure = {
        'labeled/harmful': '19zZlOUtRaJhFx3DE80TIkWXdBTu8fAI_',
        'labeled/non_harmful': '1boSUsmN_Cg_thbz9nG9-bQvRG5PJSq2b',
        'unlabeled': '1vUDBsvBKCLwqJZ0S6KLAMGVnViOYTXit'
    }

    image_df = create_image_dataframe(folder_structure)
    print(image_df.head())  # Display the first few rows of the DataFrame
    image_df.to_csv('image_urls.csv', index=False)
