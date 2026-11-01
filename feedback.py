"""Private Google feedback bridge; also usable from the training notebook."""

import base64
import hashlib
import io
import logging
import os
from pathlib import Path
import re
import tomllib
from typing import Any, NoReturn
from urllib.parse import urlparse
import uuid

import numpy as np
from PIL import Image
import requests

ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger("mnist.feedback")
ERROR_HINTS = {
    "LOCAL_CONFIG": "Check GOOGLE_SCRIPT_URL and FEEDBACK_SECRET in Streamlit Secrets; use the /exec deployment URL.",
    "AUTH_FAILED": "FEEDBACK_SECRET must match the Apps Script property; also check you deployed the correct script.",
    "SCRIPT_CONFIG": "Set SPREADSHEET_ID and FOLDER_ID in Apps Script Project Settings → Script Properties, using IDs, not names or full URLs.",
    "SHEET_ACCESS": "The deploying Google account cannot open that spreadsheet. Check its ID, Google Sheets format, and account permissions.",
    "SHEET_HEADERS": "The FIRST worksheet must have Timestamp, Predicted, Correct, Image URL, Submission ID, File ID in A1:F1, exactly.",
    "SHEET_WRITE": "The deploying account cannot write to the worksheet. Check editing access, protected ranges and Sheets quotas.",
    "DRIVE_ACCESS": "The deploying Google account cannot open the folder. Check FOLDER_ID, account access and Drive authorization.",
    "DRIVE_WRITE": "The deploying account cannot create/trash a file in the folder. Check write permission, Drive authorization, storage and quotas.",
    "INVALID_REQUEST": "The bridge rejected request data. Deploy the latest Code.gs and check labels/image format.",
    "UNKNOWN_ACTION": "Redeploy the latest Code.gs as a NEW VERSION; the deployed script does not support this action.",
    "GOOGLE_REJECTED": "Google rejected the request without a recognized code. Update/redeploy Code.gs and inspect Apps Script Executions.",
    "NON_JSON": "Google returned a web page, not JSON. Check /exec URL, deployment access Anyone, Execute as Me, authorization and Workspace restrictions.",
    "BAD_RESPONSE": "Unexpected bridge response. Check that GOOGLE_SCRIPT_URL points to this project's latest deployed Code.gs.",
    "HTTP_ERROR": "Google returned an HTTP error. Check deployment access, authorization, URL and service availability; see server logs for status.",
    "TIMEOUT": "Google did not respond in time. Check Apps Script Executions and quotas; retrying a submission uses the same submission ID.",
    "NETWORK": "The server could not reach Google. Check network access and retry; see server logs for the failure category.",
}


class FeedbackError(ValueError):
    """Safe to show publicly: no raw Google responses, credentials or resource IDs."""

    def __init__(self, code: str, request_id: str):
        self.code = code if code in ERROR_HINTS else "GOOGLE_REJECTED"
        self.request_id = request_id
        super().__init__(f"{self.code}: {ERROR_HINTS[self.code]} Reference: {request_id}")


def diagnostics_enabled() -> bool:
    """Opt-in UI; disable again after debugging to avoid public quota consumption."""
    value = os.environ.get("FEEDBACK_DIAGNOSTICS")
    if value is not None:
        return value.lower() in ("true", "1", "yes")
    path = ROOT / ".streamlit" / "secrets.toml"
    if path.exists():
        with path.open("rb") as stream:
            return tomllib.load(stream).get("FEEDBACK_DIAGNOSTICS") is True
    import streamlit as st
    from streamlit.errors import StreamlitSecretNotFoundError
    try:
        return st.secrets.get("FEEDBACK_DIAGNOSTICS", False) is True
    except (FileNotFoundError, StreamlitSecretNotFoundError):
        return False


def credentials() -> tuple[str, str]:
    """Environment overrides local TOML; Cloud secrets are server-side only."""
    values = {}
    path = ROOT / ".streamlit" / "secrets.toml"
    if path.exists():
        with path.open("rb") as stream:
            values.update(tomllib.load(stream))
    if not all(values.get(key) or os.environ.get(key)
               for key in ("GOOGLE_SCRIPT_URL", "FEEDBACK_SECRET")):
        import streamlit as st
        from streamlit.errors import StreamlitSecretNotFoundError
        try:
            values.update(dict(st.secrets))
        except (FileNotFoundError, StreamlitSecretNotFoundError):
            pass
    url = os.environ.get("GOOGLE_SCRIPT_URL", values.get("GOOGLE_SCRIPT_URL", ""))
    secret = os.environ.get("FEEDBACK_SECRET", values.get("FEEDBACK_SECRET", ""))
    parsed = urlparse(url)
    if (parsed.scheme != "https" or parsed.netloc != "script.google.com"
            or not parsed.path.startswith("/macros/s/")
            or not parsed.path.endswith("/exec") or parsed.query or parsed.fragment
            or not secret):
        raise ValueError("Configure GOOGLE_SCRIPT_URL (/exec HTTPS URL) and FEEDBACK_SECRET.")
    return url, secret


def _post(action: str, **fields) -> dict[str, Any]:
    request_id = str(uuid.uuid4())

    def fail(code: str, **details) -> NoReturn:
        # Never log request/response bodies, URLs, raw exception messages, secrets or image data.
        LOGGER.error("Google feedback action=%s code=%s request_id=%s details=%s",
                     action, code, request_id, details)
        raise FeedbackError(code, request_id) from None

    try:
        url, secret = credentials()
    except Exception as error:
        fail("LOCAL_CONFIG", exception_type=type(error).__name__)
    try:
        response = requests.post(
            url, json={"secret": secret, **fields, "action": action, "request_id": request_id},
            timeout=(10, 90)
        )
        response.raise_for_status()
    except requests.Timeout:
        fail("TIMEOUT")
    except requests.HTTPError as error:
        status = error.response.status_code if error.response is not None else None
        fail("HTTP_ERROR", http_status=status)
    except requests.RequestException as error:
        fail("NETWORK", exception_type=type(error).__name__)
    try:
        result = response.json()
    except ValueError:
        # HTML can contain account details/tokens: do not log any response snippet.
        fail("NON_JSON")
    if not isinstance(result, dict) or not isinstance(result.get("success"), bool):
        fail("BAD_RESPONSE")
    if result["success"] is not True:
        code = result.get("code")
        if not isinstance(code, str) or code not in ERROR_HINTS:
            code = "AUTH_FAILED" if result.get("error") == "Unauthorized" else "GOOGLE_REJECTED"
        fail(code)
    LOGGER.info("Google feedback action=%s success request_id=%s", action, request_id)
    return result


def check_google_connection() -> dict[str, Any]:
    """Check configuration and write permissions without appending a feedback row."""
    return _post("health")


def _digit(value):
    # Reject floats such as 2.5, missing labels and strings containing formulas.
    text = str(value).strip()
    if not re.fullmatch(r"[0-9](?:\.0)?", text):
        raise ValueError(f"Invalid digit label: {text!r}")
    return int(float(text))


def submit_feedback(
    normalized_image: np.ndarray, predicted: int, correct: int, submission_id: str
) -> dict[str, Any]:
    image = np.asarray(normalized_image).squeeze()
    if (image.shape != (28, 28) or not np.isfinite(image).all()
            or image.min() < 0 or image.max() > 1 or image.max() == 0):
        raise ValueError("Feedback must be a nonblank normalized 28x28 model input.")
    predicted, correct = _digit(predicted), _digit(correct)
    if predicted == correct:
        raise ValueError("The correction must differ from the prediction.")
    pixels = np.rint(image * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return _post(
        "submit", image=base64.b64encode(buffer.getvalue()).decode("ascii"),
        predicted=predicted, correct=correct, submission_id=submission_id
    )


def _decode_png(content):
    if len(content) > 10000:
        raise ValueError("Feedback PNG exceeds the size limit.")
    with Image.open(io.BytesIO(content)) as image:
        if image.format != "PNG" or image.size != (28, 28) or image.mode != "L":
            raise ValueError("Expected an unmodified 28x28 grayscale PNG.")
        pixels = np.array(image, dtype=np.uint8)
    if not pixels.any():
        raise ValueError("Blank feedback image.")
    return pixels


def _file_id(row):
    if row.get("File ID"):
        return str(row["File ID"]).strip()
    url = str(row.get("Image URL", ""))
    match = re.search(r"/d/([A-Za-z0-9_-]+)", url)
    if not match:
        match = re.search(r"[?&]id=([A-Za-z0-9_-]+)", url)
    if not match:
        raise ValueError("Row needs File ID, a Google Drive Image URL, or Image Path.")
    return match.group(1)


def load_feedback(config_path=None):
    """Return uint8 images and corrected labels. Disabled unless configured.

    Local CSV/XLSX columns: Correct, plus File ID/Image URL or Image Path.
    Private Drive files are read by the authenticated bridge, not public URLs.
    """
    config_path = Path(config_path or ROOT / "feedback-config.toml")
    empty = (np.empty((0, 28, 28), dtype=np.uint8), np.empty(0, dtype=np.int64))
    if not config_path.exists():
        return empty
    with config_path.open("rb") as stream:
        config = tomllib.load(stream).get("feedback", {})
    if not config.get("enabled", False):
        return empty
    limit = int(config.get("max_samples", 2000))
    if not 1 <= limit <= 10000:
        raise ValueError("max_samples must be between 1 and 10000.")
    source = config.get("source", "google")
    base_dir = config_path.resolve().parent
    if source == "google":
        def rows():
            offset = 0
            while offset < limit:
                result = _post("export", offset=offset, limit=min(50, limit - offset))
                yield from result["rows"]
                next_offset = result.get("next_offset")
                if next_offset is None:
                    break
                if not isinstance(next_offset, int) or next_offset <= offset:
                    raise ValueError("Invalid export pagination.")
                offset = next_offset
        records = rows()
    elif source in ("csv", "xlsx"):
        import pandas as pd
        path = Path(config["path"])
        if not path.is_absolute():
            path = base_dir / path
        table = (pd.read_csv(str(path), keep_default_na=False, iterator=False, chunksize=None) if source == "csv"
                 else pd.read_excel(path, keep_default_na=False, engine="openpyxl"))
        table.columns = table.columns.str.strip()
        if "Correct" not in table.columns:
            raise ValueError("Feedback spreadsheet needs a Correct column.")
        records = table.head(limit).to_dict("records")
        base_dir = path.resolve().parent
    else:
        raise ValueError("Feedback source must be google, csv or xlsx.")

    images, labels, seen = [], [], {}
    for row in records:
        label = _digit(row["Correct"])
        if row.get("image"):
            content = base64.b64decode(row["image"], validate=True)
        elif row.get("Image Path"):
            path = Path(str(row["Image Path"]))
            content = (path if path.is_absolute() else base_dir / path).read_bytes()
        else:
            result = _post("image", file_id=_file_id(row))
            content = base64.b64decode(result["image"], validate=True)
        pixels = _decode_png(content)
        digest = hashlib.sha256(pixels.tobytes()).hexdigest()
        if digest in seen:
            if seen[digest] != label:
                raise ValueError("Identical feedback images have conflicting corrected labels. Review the sheet.")
            continue
        seen[digest] = label
        images.append(pixels)
        labels.append(label)
    return (np.stack(images), np.asarray(labels, dtype=np.int64)) if images else empty


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check the private Google feedback bridge.")
    parser.add_argument("--check", action="store_true", help="Probe Sheet/Drive access (creates and trashes one test file).")
    args = parser.parse_args()
    if args.check:
        try:
            check_google_connection()
            print("Google check passed: authentication, properties, headers, Sheet write and Drive write/trash.")
        except FeedbackError as error:
            print(str(error))
            raise SystemExit(1) from None
    else:
        parser.print_help()
