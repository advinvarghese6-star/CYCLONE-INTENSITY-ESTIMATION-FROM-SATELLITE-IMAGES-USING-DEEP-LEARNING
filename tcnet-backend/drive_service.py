import os
import io
import sys
import logging
from pathlib import Path
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError

from config import SERVICE_ACCOUNT_FILE, DRIVE_FILE_ID, MODEL_LOCAL_PATH, SCOPES

logger = logging.getLogger("tcnet.drive")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def get_drive_service():
    """
    Authenticates with Google Drive v3 using the Service Account JSON credentials.
    """
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        raise FileNotFoundError(
            f"Service account file not found at: {SERVICE_ACCOUNT_FILE}. "
            "Please ensure credentials.json is placed in tcnet-backend/."
        )

    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    return service, creds


def ensure_model_downloaded() -> bool:
    """
    Checks if MODEL_LOCAL_PATH exists on disk.
    If missing, downloads it from Google Drive using the Service Account and DRIVE_FILE_ID.
    Returns True if model file is ready, False otherwise.
    """
    model_path = Path(MODEL_LOCAL_PATH)

    # 1. Check if model already exists and is non-empty
    if model_path.exists() and model_path.stat().st_size > 0:
        logger.info(f"Model checkpoint found locally at: {model_path.resolve()} ({model_path.stat().st_size} bytes)")
        return True

    logger.warning(f"Model file not found at: {model_path.resolve()}. Initiating Google Drive download...")

    # 2. Check DRIVE_FILE_ID
    if not DRIVE_FILE_ID or DRIVE_FILE_ID in ["YOUR_COPIED_FILE_ID_HERE", "YOUR_EXTRACTED_FILE_ID", ""]:
        logger.error(
            "DRIVE_FILE_ID is not configured in config.py or environment variables!\n"
            "To fix this:\n"
            "1. Upload 'baseline_cnn_v3.pt' to Google Drive.\n"
            "2. Share the file with the Service Account email:\n"
            "   website-downloader@balmy-curve-511703-v3.iam.gserviceaccount.com (Viewer role)\n"
            "3. Copy the File ID from the URL: https://drive.google.com/file/d/<FILE_ID>/view\n"
            "4. Paste it into config.py (DRIVE_FILE_ID) or set the DRIVE_FILE_ID environment variable."
        )
        return False

    # 3. Authenticate and download
    temp_path = model_path.with_suffix(".pt.download_tmp")
    try:
        service, creds = get_drive_service()
        logger.info(f"Authenticated as service account: {getattr(creds, 'service_account_email', 'unknown')}")
        logger.info(f"Requesting file ID: {DRIVE_FILE_ID}")

        request = service.files().get_media(fileId=DRIVE_FILE_ID, supportsAllDrives=True)

        with open(temp_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request, chunksize=1024 * 1024 * 5)  # 5 MB chunks
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    progress_pct = int(status.progress() * 100)
                    logger.info(f"Downloading model: {progress_pct}% complete ({status.resumable_progress} bytes)")

        # Download completed, rename temp to final destination
        if temp_path.exists():
            if model_path.exists():
                model_path.unlink()
            temp_path.rename(model_path)
            logger.info(f"Model successfully downloaded and saved to: {model_path.resolve()}")
            return True

    except HttpError as http_err:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        if http_err.resp.status == 404:
            logger.error(
                f"Google Drive API 404 Not Found for file ID: '{DRIVE_FILE_ID}'.\n"
                "Please verify that the File ID is correct and that the file has not been deleted."
            )
        elif http_err.resp.status == 403:
            logger.error(
                f"Google Drive API 403 Forbidden for file ID: '{DRIVE_FILE_ID}'.\n"
                "The service account lacks permission to view this file.\n"
                "SOLUTION: Open Google Drive, right-click the file -> Share -> Add:\n"
                "website-downloader@balmy-curve-511703-v3.iam.gserviceaccount.com\n"
                "and set permissions to 'Viewer'."
            )
        else:
            logger.error(f"Google Drive HTTP Error ({http_err.resp.status}): {http_err}")
        return False

    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        logger.error(f"Unexpected error while downloading model from Google Drive: {exc}")
        return False

    return False


if __name__ == "__main__":
    success = ensure_model_downloaded()
    if success:
        print("Model verification succeeded.")
    else:
        print("Model verification failed or needs setup.")
