# Google feedback collection and retraining

## What is saved?

Below **Prediction result**, the bold red **Wrong?** button opens a correction
form. Users choose the intended digit and consent before uploading. The server
sends the exact normalized model input, rounded back to an 8-bit, **28×28 grayscale
PNG**, plus predicted/correct labels and a random submission ID. No name or
original 280×280 canvas is collected. A successful submission disables the button
for that prediction. Retries reuse the ID so they do not append duplicate rows.
Changing the drawing invalidates the old prediction and its correction form.

MNIST supplies **60,000 training images**, of which this notebook reserves **6,000
for validation**, and **10,000 independent test images**. It is not a 6,000-image
test dataset. Feedback collection does not update the deployed model immediately.

## 1. Google Sheet and Drive folder

1. Create a Google spreadsheet named **MNIST Feedback**.
2. In its **first worksheet**, enter these exact headers in cells A1:F1:

   | Timestamp | Predicted | Correct | Image URL | Submission ID | File ID |
   | --- | --- | --- | --- | --- | --- |

   The last two columns support duplicate protection and private image downloads.
   Do not remove, rename or reorder the headers.
3. Copy `SPREADSHEET_ID` from `https://docs.google.com/spreadsheets/d/ID/edit`.
4. Create a **private**, unshared Google Drive folder named **MNIST Feedback Images**.
5. Copy `FOLDER_ID` from `https://drive.google.com/drive/folders/ID`.
6. Keep the Sheet private too. The deploying account must own or have write access
   to both. Do not publish the Sheet or make the PNGs public.

## 2. Apps Script deployment

1. From the Sheet, choose **Extensions → Apps Script**.
2. Replace the default code with the contents of `google-apps-script/Code.gs`.
3. Open **Project Settings → Script Properties → Add script property**:

   | Property | Value |
   | --- | --- |
   | `FOLDER_ID` | Drive folder ID from step 1 |
   | `SPREADSHEET_ID` | Spreadsheet ID from step 1 |
   | `FEEDBACK_SECRET` | A long random secret (at least 32 random bytes) |

   Generate a secret with `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
   Copy it privately; do not paste it into source files, URLs, or screenshots.
4. Choose **Deploy → New deployment → Web app**.
5. Set **Execute as: Me**, **Who has access: Anyone**. Authorize Drive/Sheets access.
   Workspace administrators may prohibit anonymous web apps; if so, this bridge
   requires an allowed account or a different backend.
6. Copy the deployment URL ending in **`/exec`**, not `/dev`.
7. After changing script code, choose **Manage deployments → Edit → New version →
   Deploy** to update that URL. Script Properties can be changed without committing
   credentials to the script.

Opening the URL in a browser intentionally does **not** list images: GET requests
are denied. The Python server sends the secret in authenticated POST bodies for
`submit`, paginated `export`, and single `image` requests. Drive files never need
public sharing permissions or OAuth credentials in Python.

## 3. Streamlit secrets and installation

From the project folder in PowerShell:

```powershell
python -m pip install -r requirements.txt
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
```

Edit **`.streamlit/secrets.toml`**:

```toml
GOOGLE_SCRIPT_URL = "https://script.google.com/macros/s/YOUR_DEPLOYMENT_ID/exec"
FEEDBACK_SECRET = "THE_SAME_SECRET_AS_THE_SCRIPT_PROPERTY"
```

For **Streamlit Community Cloud**, paste those same two entries into **App settings
→ Secrets**. Do not provide the folder or spreadsheet ID to Streamlit: those live
 in Apps Script Properties only. Google credentials are required only for feedback uploads or diagnostic checks,
so prediction still works without Google configuration. Restart after configuration
changes. Launch with `python -m streamlit run app.py`.

The notebook reads local `.streamlit/secrets.toml` too. For a remote notebook or
CI, set the environment variables `GOOGLE_SCRIPT_URL` and `FEEDBACK_SECRET` using
that platform's secret store; environment values override local TOML. Python 3.11+
is required for built-in TOML support. Keep secrets server-side, never in canvas
JavaScript or query strings. `.gitignore` excludes the real secrets file, local
feedback configuration and `feedback-data/`; examples are safe to commit.

## 4. Verify collection

1. Predict a drawing, select **Wrong?**, choose a different digit, consent and submit.
2. Confirm a row in the Sheet and a PNG in the Drive folder. The PNG should match
   the app's processed input, not the original canvas.
3. Submit without consent or without a label: nothing should be uploaded.
4. Change/clear the canvas: the previous correction must disappear.
5. If upload fails, check matching secrets, `/exec` URL, Script Properties, exact
   headers, account permissions and the Apps Script **Executions** panel. Errors
   now show safe codes and a request reference. There is no offline upload queue.

### Drive/Sheets permissions and troubleshooting

**How permissions work:** the Web app runs as the Google account that deployed it
when **Execute as: Me** is selected. That account must authorize Apps Script to use
Drive and Sheets and must have editing access to the target spreadsheet and folder.
Streamlit does not need Google Drive sharing access, an OAuth token or a service
account. Keep the folder and Sheet private. If they belong to another Google
account, share them privately with the deploying account with editing access.

1. Paste the updated `google-apps-script/Code.gs` into the Apps Script editor and save.
2. Select **testSetup** in the function dropdown next to **Run**, then click **Run**.
   Do not run `doPost` directly: it expects an HTTP event.
3. If asked, select **Review permissions**, choose the intended Google account, and
   authorize the Drive/Sheets scopes of your own script. Do not bypass a Workspace
   administrator's policy; ask the administrator if authorization is blocked.
4. Read the editor's **Execution log** or **Executions → latest execution → logs**.
   A result with `success: true` confirms properties, first-tab headers, Sheet write
   access and Drive create/trash permission. The check rewrites the existing A1
   header, creates one tiny `.txt` file in the folder, then trashes it. It does not
   append a feedback row. A failed trash operation may leave that tiny test file.
5. Update the actual Web app: **Deploy → Manage deployments → Edit (pencil) →
   Version: New version → Deploy**. Confirm **Execute as: Me** and **Anyone** access.
   Saving editor code or running `testSetup` alone does NOT update the deployed `/exec`
   version. Reauthorize when prompted. If making a new deployment, update its URL
   in Streamlit Secrets; editing the existing deployment preserves its URL.
6. Commit/push the changed Python files to the Streamlit deployment branch too.

**Streamlit connection check:** temporarily add `FEEDBACK_DIAGNOSTICS = true`
(a TOML boolean, not a quoted string) to **Streamlit Cloud → App settings → Secrets**
or local `.streamlit/secrets.toml`. Restart if needed. Expand **Google feedback
diagnostics** above the canvas and click **Check Google connection**. This runs the
same write probe through the real deployed endpoint. Set the flag back to `false`
after debugging, since all visitors can use the button while it is enabled and
each click consumes Google quota. No secret values are displayed in the panel.

For local testing, `python feedback.py --check` runs the same authenticated check
using existing server-side secrets/environment variables. It never prints those
values and does not require enabling the UI flag.

**Where to find server logs:** in Streamlit Cloud open **Manage app → logs**. Locally
read the terminal running Streamlit. Look for `Google feedback action=... code=...
request_id=...`. In **Apps Script → Executions**, open the matching POST execution
and find the same `request_id`. You can share the **code and reference**, not secrets
or complete request/response bodies. Raw Google error messages and HTML pages are
deliberately not logged because they may contain account/resource identifiers.

| Code | What to check |
| --- | --- |
| `LOCAL_CONFIG` | Streamlit `GOOGLE_SCRIPT_URL` / `FEEDBACK_SECRET`, valid HTTPS `/exec` URL |
| `AUTH_FAILED` | Secret matches Apps Script `FEEDBACK_SECRET`; correct deployed project |
| `SCRIPT_CONFIG` | Script Properties contain actual `SPREADSHEET_ID` / `FOLDER_ID`, not names or full URLs |
| `SHEET_ACCESS` | Correct Google Sheets ID, deploying account access; convert `.xlsx` files to Google Sheets |
| `SHEET_HEADERS` | Exact A1:F1 headers on the **first worksheet tab** |
| `SHEET_WRITE` | Editing permission, protected A1/worksheet, Sheet quota |
| `DRIVE_ACCESS` | Correct folder ID, deploying account's folder access and Drive authorization |
| `DRIVE_WRITE` | Create/trash permission, authorization, Drive storage/quota |
| `NON_JSON` / `HTTP_ERROR` | Login/permission HTML, wrong URL, anonymous access, deployment authorization, Workspace restrictions |
| `UNKNOWN_ACTION` / `GOOGLE_REJECTED` | Old/wrong deployed Code.gs; deploy a new version and inspect Google Executions |
| `TIMEOUT` / `NETWORK` | Google quotas/executions, service/network availability; safe to retry a submission |

If the editor's `testSetup` succeeds but Streamlit fails, focus on the deployed
version, `/exec` URL, anonymous access and matching secret. If Streamlit's health
check passes but submissions fail, retry a correction and report its new code and
reference; the failure is likely in validation, file creation or row appending.

Offline regression tests: `python -m unittest discover -s tests -v` and
`node --test tests/test_google_bridge.js` (Node is needed only for the mocked script
tests, not for Streamlit deployment).

## 5. Optional training configuration

Copy `feedback-config.example.toml` to **`feedback-config.toml`**. Missing config
or `enabled = false` keeps the original MNIST-only workflow.

### Option A — Pull directly from private Google Sheets and Drive (recommended)

```toml
[feedback]
enabled = true
source = "google"
max_samples = 2000
```

Rerun `train-model.ipynb` from the beginning. `feedback.load_feedback()` downloads
metadata **and PNGs** through the authenticated bridge, 50 rows at a time. It uses
the **Correct** label, never the prediction. `max_samples` limits examined rows
in sheet order (1–10,000). Duplicate image pixels with the same label are removed;
conflicting labels, invalid digits, missing images and invalid PNGs stop training
instead of silently poisoning the dataset.

### Option B — Export a CSV or Excel spreadsheet

In Google Sheets choose **File → Download → Comma-separated values (.csv)** or
**Microsoft Excel (.xlsx)**. Put the downloaded file inside `feedback-data/`.

```toml
[feedback]
enabled = true
source = "csv" # or "xlsx"
path = "feedback-data/MNIST Feedback.csv" # use .xlsx for Excel
max_samples = 2000
```

The table needs **Correct** and either **File ID** or **Image URL**. The standard
six-column export works unchanged. Metadata is local, but private Drive PNGs still
come through the bridge and require the same secrets. Editing `Correct` in a local
export uses that local label for training; it does not update Google Sheets.

For completely offline training, download the PNGs from Drive yourself and add
an **Image Path** column to the CSV/Excel file, such as `images/digit_example.png`.
Paths are relative to the CSV/Excel directory unless absolute. No Google credentials
are needed when every row has a local Image Path. Preserve the original 28×28
grayscale PNGs; the loader deliberately does not resize or recrop them.

### Review and deploy a new iteration

- Review labels/drawings before training; the loader has no automatic review flag.
  Use a reviewed CSV/Excel subset, or clean the live Sheet before direct loading.
- Feedback is added **only to training**, after separating 54,000 MNIST training
  samples and the fixed 6,000 MNIST validation samples. Normalization is unchanged.
  The 10,000-image MNIST test set is never appended to training.
- This notebook trains the three CNN candidates from scratch and selects the best
  using MNIST validation accuracy. It does not automatically fine-tune the live model.
- Compare against a baseline and keep a separate set of real drawings that are
  **never used for training**. MNIST accuracy alone cannot show that new handwriting
  improved. Avoid duplicate drawings across your training and evaluation sets.
- Small feedback sets may have little effect; too much biased feedback may hurt
  accuracy. No improvement is guaranteed. Increase data gradually and evaluate.
- The notebook saves `best_mnist_cnn.keras`. Back up the existing model first, then
  redeploy the new file and **restart Streamlit** to clear its cached model.
- To remove feedback, delete its Sheet row **and** Drive PNG (and any local exports).
  Deleting collected data does not undo its influence in an already trained model;
  that requires retraining without the deleted samples.

## Limits and security

This is a small-experiment bridge, not a production ingestion service. Anyone can
reach the endpoint, but actions need the shared server-side secret. Rotate the
secret in both places if leaked. Public app users can still send many drawings via
the app; consent and duplicate IDs are **not rate limiting** or label verification.
Apps Script/Drive/Sheets have quotas; exports also consume quota. Add abuse controls
and proper authenticated storage for production. Avoid collecting personally
identifiable drawings, limit access to the Google account, and define a retention
policy before sharing the app widely.
