"""Test feedback interactions without a real canvas, TensorFlow or Google account."""

import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

from feedback import ROOT


class AppFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.pixels = np.zeros((280, 280, 4), dtype=np.uint8)
        self.pixels[:, :, 3] = 255
        self.pixels[50:230, 125:145, :3] = 255
        probabilities = np.full((1, 10), 0.01)
        probabilities[0, 1] = 0.91
        model = SimpleNamespace(predict=lambda *args, **kwargs: probabilities)
        tensorflow = SimpleNamespace(keras=SimpleNamespace(
            models=SimpleNamespace(load_model=lambda *args, **kwargs: model)))
        canvas = SimpleNamespace(st_canvas=lambda **kwargs: SimpleNamespace(image_data=self.pixels))
        modules = patch.dict(sys.modules, {"tensorflow": tensorflow, "streamlit_drawable_canvas": canvas})
        modules.start()
        self.addCleanup(modules.stop)
        upload = patch("feedback.submit_feedback", return_value={"success": True})
        self.upload = upload.start()
        self.addCleanup(upload.stop)
        self.app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
        self.assert_no_errors()
        self.app.button[0].click().run()
        self.assert_no_errors()
        self.app.button(key="wrong_button").click().run()
        self.assert_no_errors()

    def assert_no_errors(self):
        self.assertFalse(self.app.exception, [error.message for error in self.app.exception])

    def submit_form(self, consent: bool = True, label: int | None = 7):
        if label is not None:
            self.app.selectbox[0].select(label)
        if consent:
            self.app.checkbox[0].check()
        next(button for button in self.app.button if button.label == "Submit correction").click().run()
        self.assert_no_errors()

    def test_requires_label_and_consent(self):
        self.submit_form(consent=False, label=None)
        self.upload.assert_not_called()
        self.submit_form(consent=False)
        self.upload.assert_not_called()

    def test_success_saves_current_input_and_disables_wrong_button(self):
        self.submit_form()
        self.upload.assert_called_once()
        self.assertEqual(self.upload.call_args.args[0].shape, (28, 28))
        self.assertEqual(self.upload.call_args.args[1:3], (1, 7))
        self.assertTrue(self.app.session_state["feedback_saved"])
        self.assertTrue(self.app.button(key="wrong_button").disabled)

    def test_failed_upload_retries_same_submission_id(self):
        self.upload.side_effect = [TimeoutError("offline"), {"success": True}]
        self.submit_form()
        self.assertFalse(self.app.session_state["feedback_saved"])
        submission_id = self.upload.call_args.args[3]
        self.submit_form()
        self.assertTrue(self.app.session_state["feedback_saved"])
        self.assertEqual(self.upload.call_args.args[3], submission_id)

    def test_drawing_change_invalidates_old_form(self):
        self.pixels[100, 100, :3] = 255
        self.app.run()
        self.assert_no_errors()
        self.assertIsNone(self.app.session_state["prediction"])
        self.assertFalse(self.app.session_state["show_feedback"])
        self.assertFalse(self.app.session_state["feedback_saved"])
        self.upload.assert_not_called()


if __name__ == "__main__":
    unittest.main()

