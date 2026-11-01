import base64
import io
import json
from pathlib import Path
import tempfile
from typing import Any
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import requests

import feedback


def png(value=128):
    stream = io.BytesIO()
    Image.fromarray(np.full((28, 28), value, dtype=np.uint8)).save(stream, format="PNG")
    return stream.getvalue()


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def config(self, source="google", **options):
        path = self.root / "feedback-config.toml"
        values = {"enabled": True, "source": source, **options}
        text = "[feedback]\n" + "\n".join(
            f"{key} = {json.dumps(value)}" for key, value in values.items()
        )
        path.write_text(text, encoding="utf-8")
        return path

    def test_unconfigured_training_is_optional(self):
        images, labels = feedback.load_feedback(self.root / "absent.toml")
        self.assertEqual(images.shape, (0, 28, 28))
        self.assertEqual(len(labels), 0)

    def test_disabled_training_does_not_connect(self):
        with patch.object(feedback, "_post") as post:
            feedback.load_feedback(self.config(enabled=False))
            post.assert_not_called()

    def test_upload_preserves_model_pixels_and_id(self):
        pixels = np.arange(784, dtype=np.uint16).reshape(28, 28).astype(np.uint8)
        image = pixels.astype(np.float32) / 255
        with patch.object(feedback, "_post", return_value={"success": True}) as post:
            feedback.submit_feedback(image, 2, 7, "test-id")
            feedback.submit_feedback(image, 2, 7, "test-id")
        payload = post.call_args.kwargs
        actual = feedback._decode_png(base64.b64decode(payload["image"]))
        np.testing.assert_array_equal(actual, pixels)
        self.assertEqual(payload["correct"], 7)
        self.assertEqual(payload["submission_id"], "test-id")
        self.assertEqual(post.call_args_list[0], post.call_args_list[1])

    def test_invalid_input_is_not_uploaded(self):
        bad_images = [np.zeros((28, 28)), np.ones((10, 10)),
                      np.full((28, 28), np.nan), np.full((28, 28), 2.0)]
        with patch.object(feedback, "_post") as post:
            for image in bad_images:
                with self.subTest(shape=image.shape):
                    with self.assertRaises(ValueError):
                        feedback.submit_feedback(image, 2, 7, "id")
            post.assert_not_called()

    def test_invalid_labels_and_identical_correction(self):
        for value in [-1, 10, 2.5, "=SUM(A1)", "", None, True]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                feedback._digit(value)
        with self.assertRaises(ValueError):
            feedback.submit_feedback(np.ones((28, 28)), 2, 2, "id")

    def test_post_uses_body_secret_and_timeout(self):
        response = Mock()
        response.json.return_value = {"success": True}
        with patch.object(feedback, "credentials", return_value=("https://example.test", "private")), \
                patch("requests.post", return_value=response) as post:
            feedback._post("export", offset=0, limit=50)
        self.assertEqual(post.call_args.kwargs["json"]["secret"], "private")
        self.assertEqual(post.call_args.kwargs["timeout"], (10, 90))
        self.assertRegex(post.call_args.kwargs["json"]["request_id"], r"^[a-f0-9-]{36}$")
        response.raise_for_status.assert_called_once()

    def test_service_failure_is_not_success(self):
        response = Mock()
        response.json.return_value = {"success": False, "error": "Unauthorized"}
        with patch.object(feedback, "credentials", return_value=("https://example.test", "private")), \
                patch("requests.post", return_value=response), \
                self.assertRaises(ValueError):
            feedback._post("submit")

    def test_credentials_use_server_environment(self):
        url = "https://script.google.com/macros/s/deployment/exec"
        with patch.object(feedback, "ROOT", self.root), patch.dict(
                feedback.os.environ, {"GOOGLE_SCRIPT_URL": url, "FEEDBACK_SECRET": "private"}):
            self.assertEqual(feedback.credentials(), (url, "private"))

    def test_sanitized_google_errors_are_actionable(self):
        response = Mock()
        response.json.return_value = {
            "success": False, "code": "DRIVE_WRITE", "error": "SECRET_AND_PRIVATE_FOLDER_ID"
        }
        with patch.object(feedback, "credentials", return_value=("https://example.test/private-id", "PRIVATE_SECRET")), \
                patch("requests.post", return_value=response) as post, \
                self.assertLogs("mnist.feedback", level="ERROR") as logs, \
                self.assertRaises(feedback.FeedbackError) as caught:
            feedback._post("health")
        self.assertEqual(caught.exception.code, "DRIVE_WRITE")
        self.assertEqual(caught.exception.request_id, post.call_args.kwargs["json"]["request_id"])
        output = str(caught.exception) + " ".join(logs.output)
        for sensitive in ["PRIVATE_SECRET", "private-id", "SECRET_AND_PRIVATE_FOLDER_ID"]:
            self.assertNotIn(sensitive, output)

    def test_transport_failure_categories_do_not_leak_messages(self):
        http_response = Mock(status_code=403)
        cases = [
            (requests.Timeout("LEAKED_SECRET"), "TIMEOUT"),
            (requests.ConnectionError("LEAKED_SECRET"), "NETWORK"),
            (requests.HTTPError("LEAKED_SECRET", response=http_response), "HTTP_ERROR"),
        ]
        for exception, code in cases:
            with self.subTest(code=code), \
                    patch.object(feedback, "credentials", return_value=("https://example.test", "secret")), \
                    patch("requests.post", side_effect=exception), \
                    self.assertLogs("mnist.feedback", level="ERROR") as logs, \
                    self.assertRaises(feedback.FeedbackError) as caught:
                feedback._post("submit")
            self.assertEqual(caught.exception.code, code)
            self.assertNotIn("LEAKED_SECRET", str(caught.exception) + " ".join(logs.output))
            if code == "HTTP_ERROR":
                self.assertIn("403", " ".join(logs.output))

    def test_html_and_invalid_json_response_categories(self):
        for payload, expected in [(None, "NON_JSON"), (["private data"], "BAD_RESPONSE"),
                                  ({"success": False, "error": "private data"}, "GOOGLE_REJECTED")]:
            response = Mock()
            if payload is None:
                response.json.side_effect = ValueError("PRIVATE_HTML_TOKEN")
            else:
                response.json.return_value = payload
            with self.subTest(expected=expected), \
                    patch.object(feedback, "credentials", return_value=("https://example.test", "secret")), \
                    patch("requests.post", return_value=response), \
                    self.assertLogs("mnist.feedback", level="ERROR") as logs, \
                    self.assertRaises(feedback.FeedbackError) as caught:
                feedback._post("health")
            self.assertEqual(caught.exception.code, expected)
            self.assertNotIn("private data", " ".join(logs.output))
            self.assertNotIn("PRIVATE_HTML_TOKEN", str(caught.exception))

    def test_bad_configuration_is_logged_without_contents(self):
        with patch.object(feedback, "credentials", side_effect=ValueError("PRIVATE_SECRET")), \
                self.assertLogs("mnist.feedback", level="ERROR") as logs, \
                self.assertRaises(feedback.FeedbackError) as caught:
            feedback._post("health")
        self.assertEqual(caught.exception.code, "LOCAL_CONFIG")
        self.assertNotIn("PRIVATE_SECRET", str(caught.exception) + " ".join(logs.output))

    def test_health_check_uses_authenticated_transport(self):
        with patch.object(feedback, "_post", return_value={"success": True}) as post:
            feedback.check_google_connection()
        post.assert_called_once_with("health")

    def test_diagnostics_are_opt_in(self):
        with patch.dict(feedback.os.environ, {"FEEDBACK_DIAGNOSTICS": "false"}):
            self.assertFalse(feedback.diagnostics_enabled())
        with patch.dict(feedback.os.environ, {"FEEDBACK_DIAGNOSTICS": "true"}):
            self.assertTrue(feedback.diagnostics_enabled())

    def test_credentials_reject_non_google_endpoint(self):
        with patch.object(feedback, "ROOT", self.root), patch.dict(
                feedback.os.environ, {"GOOGLE_SCRIPT_URL": "http://example.com/exec", "FEEDBACK_SECRET": "private"}), \
                self.assertRaises(ValueError):
            feedback.credentials()

    def test_google_pagination_and_correct_labels(self):
        first = {"rows": [{"Correct": 7, "Predicted": 2,
                            "image": base64.b64encode(png(128)).decode()}], "next_offset": 1}
        second = {"rows": [{"Correct": 9, "Predicted": 1,
                             "image": base64.b64encode(png(129)).decode()}], "next_offset": None}
        with patch.object(feedback, "_post", side_effect=[first, second]) as post:
            images, labels = feedback.load_feedback(self.config())
        self.assertEqual(images.shape, (2, 28, 28))
        np.testing.assert_array_equal(labels, [7, 9])
        self.assertEqual(post.call_args.kwargs["offset"], 1)

    def test_google_deduplicates_and_rejects_conflicts(self):
        row = {"Correct": 7, "image": base64.b64encode(png()).decode()}
        with patch.object(feedback, "_post", return_value={"rows": [row, row], "next_offset": None}):
            images, labels = feedback.load_feedback(self.config())
            self.assertEqual(len(labels), 1)
        with patch.object(feedback, "_post", return_value={
                "rows": [row, {**row, "Correct": 2}], "next_offset": None}), \
                self.assertRaises(ValueError):
            feedback.load_feedback(self.config())

    def test_offline_csv_has_no_google_dependency(self):
        (self.root / "digit.png").write_bytes(png())
        (self.root / "labels.csv").write_text("Correct,Image Path\n7,digit.png\n", encoding="utf-8")
        with patch.object(feedback, "_post") as post:
            images, labels = feedback.load_feedback(self.config("csv", path="labels.csv"))
            post.assert_not_called()
        self.assertEqual(int(labels[0]), 7)
        self.assertEqual(images.dtype, np.uint8)

    def test_csv_private_drive_download(self):
        (self.root / "labels.csv").write_text(
            "Correct,Image URL\n7,https://drive.google.com/file/d/private_id/view\n", encoding="utf-8")
        with patch.object(feedback, "_post", return_value={"image": base64.b64encode(png()).decode()}) as post:
            feedback.load_feedback(self.config("csv", path="labels.csv"))
        post.assert_called_once_with("image", file_id="private_id")

    def test_excel_local_images(self):
        import pandas as pd
        (self.root / "digit.png").write_bytes(png())
        pd.DataFrame({"Correct": [7], "Image Path": ["digit.png"]}).to_excel(
            self.root / "labels.xlsx", index=False, engine="openpyxl")
        images, labels = feedback.load_feedback(self.config("xlsx", path="labels.xlsx"))
        self.assertEqual(images.shape, (1, 28, 28))
        self.assertEqual(int(labels[0]), 7)

    def test_limits_and_invalid_png(self):
        for limit in [0, 10001]:
            with self.assertRaises(ValueError):
                feedback.load_feedback(self.config(max_samples=limit))
        stream = io.BytesIO()
        Image.new("RGB", (28, 28), "white").save(stream, format="PNG")
        for content in [b"not a PNG", stream.getvalue(), png(0)]:
            with self.assertRaises((ValueError, OSError)):
                feedback._decode_png(content)

    def test_notebook_feedback_stays_out_of_validation(self):
        notebook = json.loads((feedback.ROOT / "train-model.ipynb").read_text(encoding="utf-8"))
        sources = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"]
                   for c in notebook["cells"] if c["cell_type"] == "code"]
        training = next(source for source in sources if "from feedback import load_feedback" in source)
        setup = training.split("# Reserve MNIST")[1].split("filter_options =")[0]
        namespace: dict[str, Any] = {"np": np, "X_train_cnn": np.zeros((100, 28, 28, 1)),
                     "y_train": np.arange(100), "load_feedback": lambda: (np.full((2, 28, 28), 255), np.array([7, 9]))}
        exec("# Reserve MNIST" + setup, namespace)
        self.assertEqual(len(namespace["y_training"]), 92)
        np.testing.assert_array_equal(namespace["y_validation"], np.arange(90, 100))
        self.assertEqual(float(namespace["X_training"][-1].max()), 1.0)
        self.assertIn("validation_data=(X_validation, y_validation)", training)
        for source in sources:
            compile(source, "notebook_cell", "exec")


if __name__ == "__main__":
    unittest.main()


