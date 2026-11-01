// Deploy as a Web app: Execute as Me; access Anyone.
// Set FOLDER_ID, SPREADSHEET_ID, FEEDBACK_SECRET in Script Properties.
const HEADERS = ["Timestamp", "Predicted", "Correct", "Image URL", "Submission ID", "File ID"];

function jsonResponse(value) {
  return ContentService.createTextOutput(JSON.stringify(value))
    .setMimeType(ContentService.MimeType.JSON);
}

function digit(value) {
  if (!Number.isInteger(value) || value < 0 || value > 9) {
    throw new Error("Invalid digit");
  }
  return value;
}

function sheetFor(properties) {
  const sheet = SpreadsheetApp.openById(properties.SPREADSHEET_ID).getSheets()[0];
  const headers = sheet.getRange(1, 1, 1, HEADERS.length).getValues()[0];
  if (headers.some((value, index) => value !== HEADERS[index])) {
    throw new Error("Check sheet headers");
  }
  return sheet;
}

function feedbackRows(sheet) {
  const count = sheet.getLastRow() - 1;
  return count > 0 ? sheet.getRange(2, 1, count, HEADERS.length).getValues() : [];
}

function privateImage(folder, fileId) {
  if (typeof fileId !== "string" || !/^[A-Za-z0-9_-]+$/.test(fileId)) {
    throw new Error("Invalid file ID");
  }
  const file = DriveApp.getFileById(fileId);
  const parents = file.getParents();
  let allowed = false;
  while (parents.hasNext()) {
    if (parents.next().getId() === folder.getId()) allowed = true;
  }
  if (!allowed || file.isTrashed() || file.getMimeType() !== "image/png" || file.getSize() > 10000) {
    throw new Error("Invalid feedback file");
  }
  return Utilities.base64Encode(file.getBlob().getBytes());
}

function submit(data, sheet, folder) {
  const predicted = digit(data.predicted);
  const correct = digit(data.correct);
  if (predicted === correct) throw new Error("Not a correction");
  if (typeof data.submission_id !== "string" ||
      !/^[a-f0-9-]{36}$/i.test(data.submission_id)) throw new Error("Invalid submission ID");
  if (typeof data.image !== "string" || data.image.length > 14000) {
    throw new Error("Image too large");
  }
  const bytes = Utilities.base64Decode(data.image);
  const unsigned = bytes.map(value => (value + 256) % 256);
  const signature = [137, 80, 78, 71, 13, 10, 26, 10];
  // PNG signature, IHDR, 28x28, 8-bit grayscale. Python also fully decodes exports.
  if (bytes.length < 33 || bytes.length > 10000 ||
      signature.some((value, index) => unsigned[index] !== value) ||
      [73, 72, 68, 82].some((value, index) => unsigned[12 + index] !== value) ||
      [0, 0, 0, 28, 0, 0, 0, 28, 8, 0].some((value, index) => unsigned[16 + index] !== value)) {
    throw new Error("Expected a 28x28 grayscale PNG");
  }
  const lock = LockService.getScriptLock();
  lock.waitLock(30000);
  try {
    if (feedbackRows(sheet).some(row => row[4] === data.submission_id)) {
      return {success: true, duplicate: true};
    }
    const blob = Utilities.newBlob(bytes, "image/png", "digit_" + data.submission_id + ".png");
    const file = folder.createFile(blob);
    try {
      sheet.appendRow([new Date(), predicted, correct, file.getUrl(), data.submission_id, file.getId()]);
      SpreadsheetApp.flush();
    } catch (error) {
      file.setTrashed(true);
      throw error;
    }
    return {success: true};
  } finally {
    lock.releaseLock();
  }
}

function doPost(e) {
  try {
    if (!e || !e.postData || e.postData.contents.length > 18000) {
      throw new Error("Invalid request");
    }
    const data = JSON.parse(e.postData.contents);
    const properties = PropertiesService.getScriptProperties().getProperties();
    if (!properties.FEEDBACK_SECRET || data.secret !== properties.FEEDBACK_SECRET) {
      return jsonResponse({success: false, error: "Unauthorized"});
    }
    const sheet = sheetFor(properties);
    const folder = DriveApp.getFolderById(properties.FOLDER_ID);
    if (data.action === "submit") return jsonResponse(submit(data, sheet, folder));
    const rows = feedbackRows(sheet);
    if (data.action === "image") {
      // Never expose arbitrary files owned by the deploying Google account.
      if (!rows.some(row => row[5] === data.file_id)) throw new Error("Unknown image");
      return jsonResponse({success: true, image: privateImage(folder, data.file_id)});
    }
    if (data.action === "export") {
      if (!Number.isInteger(data.offset) || data.offset < 0 ||
          !Number.isInteger(data.limit) || data.limit < 1 || data.limit > 50) {
        throw new Error("Invalid pagination");
      }
      const end = Math.min(rows.length, data.offset + data.limit);
      const records = rows.slice(data.offset, end).map(row => ({
        Timestamp: row[0], Predicted: row[1], Correct: row[2],
        "Image URL": row[3], "Submission ID": row[4], "File ID": row[5],
        image: privateImage(folder, row[5])
      }));
      return jsonResponse({success: true, rows: records, next_offset: end < rows.length ? end : null});
    }
    throw new Error("Unknown action");
  } catch (error) {
    // Avoid leaking Drive IDs, account details, or credentials in public errors.
    return jsonResponse({success: false, error: "Request failed. Check configuration, labels and files."});
  }
}

function doGet() {
  return jsonResponse({success: false, error: "Use authenticated POST requests."});
}
