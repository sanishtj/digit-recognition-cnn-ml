import streamlit as st
import numpy as np
import tensorflow as tf

import uuid
import logging
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from feedback import (
    FeedbackError, check_google_connection, diagnostics_enabled, submit_feedback
)


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="wide"
)

st.title("🔢 MNIST Digit Recognition Studio")
st.caption(
    "Draw a handwritten digit, inspect the preprocessing, "
    "and see how your CNN classifies it."
)


# --------------------------------------------------
# 2. LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_mnist_cnn.keras")


model = load_model()


# --------------------------------------------------
# 3. SESSION STATE
# --------------------------------------------------

defaults = {
    "canvas_version": 0,
    "gray_image": None,
    "resized_image": None,
    "normalized_image": None,
    "prediction": None,
    "drawing_hash": None,
    "show_feedback": False,
    "feedback_saved": False,
    "feedback_id": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_pipeline():
    st.session_state.gray_image = None
    st.session_state.resized_image = None
    st.session_state.normalized_image = None
    st.session_state.prediction = None
    st.session_state.show_feedback = False
    st.session_state.feedback_saved = False
    st.session_state.feedback_id = None


def show_feedback_failure(error):
    """Only explicitly sanitized diagnostics may be shown to visitors."""
    if isinstance(error, FeedbackError):
        st.error(str(error))
    else:
        reference = str(uuid.uuid4())
        logging.getLogger("mnist.feedback").error(
            "Feedback UI failure exception_type=%s request_id=%s",
            type(error).__name__, reference
        )
        st.error(f"Unexpected feedback error. Check Streamlit server logs. Reference: {reference}")


# --------------------------------------------------
# 4. IMAGE PROCESSING FUNCTIONS
# --------------------------------------------------

def convert_to_grayscale(image_data):
    """Composite the RGBA canvas over black and convert it to grayscale."""
    rgba = Image.fromarray(image_data.astype("uint8")).convert("RGBA")

    black_background = Image.new(
        "RGBA",
        rgba.size,
        (0, 0, 0, 255)
    )

    image = Image.alpha_composite(black_background, rgba)
    return image.convert("L")


def resize_image(gray_image):
    """
    Crop the digit, scale it proportionally to fit within 20x20,
    place it on a 28x28 image, and centre it by its centre of mass.
    """
    pixels = np.array(gray_image)

    # Find pixels belonging to the white digit
    coordinates = np.argwhere(pixels > 30)

    if len(coordinates) == 0:
        raise ValueError("No digit detected. Draw a digit on the canvas first.")

    # Find digit boundaries
    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0)

    # Crop tightly around the digit
    cropped = gray_image.crop((
        x_min,
        y_min,
        x_max + 1,
        y_max + 1
    ))

    # Preserve proportions; longest dimension becomes 20 pixels
    width, height = cropped.size
    scale = 20 / max(width, height)

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    digit = cropped.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # Place digit on a 28x28 black background
    final_image = Image.new("L", (28, 28), 0)

    paste_x = (28 - new_width) // 2
    paste_y = (28 - new_height) // 2
    final_image.paste(digit, (paste_x, paste_y))

    # Calculate the centre of mass using pixel intensity as the weight
    pixels = np.array(final_image, dtype="float32")
    total_intensity = pixels.sum()

    if total_intensity > 0:
        y_indices, x_indices = np.indices(pixels.shape)

        center_x = (x_indices * pixels).sum() / total_intensity
        center_y = (y_indices * pixels).sum() / total_intensity

        shift_x = int(round(13.5 - center_x))
        shift_y = int(round(13.5 - center_y))

        shifted_image = Image.new("L", (28, 28), 0)
        shifted_image.paste(final_image, (shift_x, shift_y))
        final_image = shifted_image

    return final_image


def normalize_image(resized_image):
    """Convert grayscale pixel values from 0-255 to 0-1."""
    image_array = np.array(resized_image, dtype="float32") / 255.0
    model_input = image_array.reshape(1, 28, 28, 1)
    return image_array, model_input


def predict_digit(model_input):
    probabilities = model.predict(model_input, verbose=0)[0]
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit])

    return probabilities, predicted_digit, confidence


def run_full_pipeline(image_data):
    """Run preprocessing and prediction for the current canvas drawing."""
    gray = convert_to_grayscale(image_data)
    resized = resize_image(gray)
    normalized, model_input = normalize_image(resized)
    prediction = predict_digit(model_input)

    st.session_state.gray_image = gray
    st.session_state.resized_image = resized
    st.session_state.normalized_image = normalized
    st.session_state.prediction = prediction
    st.session_state.show_feedback = False
    st.session_state.feedback_saved = False
    st.session_state.feedback_id = str(uuid.uuid4())


# --------------------------------------------------
# 5. SIDEBAR — HOW THE CNN WORKS
# --------------------------------------------------

with st.sidebar:
    st.title("🧠 How the CNN works")
    st.caption("Your drawing goes through two main stages.")

    st.markdown("### 1. Image preprocessing")
    st.markdown(
        """
        **Your drawing**
        ↓

        **Grayscale**
        - Converts the image to one brightness channel.

        ↓

        **Crop the digit**
        - Removes empty space around your handwriting.

        ↓

        **Resize and centre**
        - Preserves the digit's proportions.
        - Fits it within 20 × 20 pixels.
        - Centres it on a 28 × 28 canvas.

        ↓

        **Normalize**
        - Changes pixel values from 0–255 to 0–1.
        """
    )

    st.divider()

    st.markdown("### 2. CNN classification")
    st.markdown(
        """
        **28 × 28 × 1 input**
        ↓

        **Convolution**
        - Learns visual patterns such as edges, curves and lines.

        ↓

        **ReLU**
        - Adds non-linearity so the network can learn complex patterns.

        ↓

        **Max pooling**
        - Reduces the feature-map size while retaining important features.

        ↓

        **Flatten**
        - Converts the feature maps into a one-dimensional list.

        ↓

        **Dense layers**
        - Use the learned features to distinguish the digits.

        ↓

        **Softmax**
        - Produces probabilities for digits 0–9.
        """
    )

    st.divider()
    st.markdown("### What to look for")
    st.caption(
        "The 28 × 28 image shown on the page is the actual image "
        "passed to the CNN. If it looks unclear, misplaced or too thick, "
        "the prediction may suffer."
    )


# --------------------------------------------------
# 6. DRAWING CANVAS AND QUICK PREDICTION
# --------------------------------------------------

if diagnostics_enabled():
    with st.expander("Google feedback diagnostics"):
        st.caption(
            "Owner debugging: checks authentication, Script Properties, first-sheet "
            "headers and Sheet/Drive write access. Creates and trashes one tiny "
            "Drive test file; no feedback row is added. Disable FEEDBACK_DIAGNOSTICS "
            "after testing—this button is visible to all visitors while enabled."
        )
        if st.button("Check Google connection", key="google_health_check"):
            try:
                with st.spinner("Checking Google Sheets and Drive…"):
                    check_google_connection()
                st.success("Google connection, headers, and Sheet/Drive write checks passed.")
            except Exception as error:
                show_feedback_failure(error)
        st.caption(
            "Streamlit Cloud: Manage app → logs. Google: Apps Script → Executions. "
            "Match the request reference shown on a failure. Do not share secrets."
        )

st.subheader("Step 1 — Draw your digit")
st.write("Draw one digit from 0 to 9 using your mouse or trackpad.")

col_canvas, col_predict, col_result = st.columns(
    [1.25, 0.55, 1.0],
    vertical_alignment="center"
)

with col_canvas:
    stroke_width = st.slider(
        "✏️ Pen thickness",
        min_value=2,
        max_value=25,
        value=14,
        step=1,
        help="Adjust the thickness of your drawing."
    )

    st.caption(
        "Tips: Draw one digit • Keep it near the centre • "
        "Avoid touching the edges • Try a natural handwriting style"
    )

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_version}",
        update_streamlit=True,
        return_image_data=True
    )

image_data = (
    canvas_result.image_data
    if canvas_result is not None
    else None
)

drawing_available = (
    image_data is not None
    and image_data.ndim == 3
    and np.any(image_data[:, :, :3] > 30)
)

# Detect a changed drawing and invalidate old processing results
if image_data is not None:
    current_hash = hash(image_data.tobytes())

    if current_hash != st.session_state.drawing_hash:
        reset_pipeline()
        st.session_state.drawing_hash = current_hash

with col_predict:
    st.markdown("### Predict")
    st.caption("Ready to classify your digit?")

    predict_clicked = st.button(
        "🔍 Predict digit",
        type="primary",
        use_container_width=True,
        disabled=not drawing_available
    )

    if st.button("🗑️ Clear canvas", use_container_width=True):
        st.session_state.canvas_version += 1
        st.session_state.drawing_hash = None
        reset_pipeline()
        st.rerun()

with col_result:
    st.markdown("### Prediction result")

    with st.container(border=True):
        if st.session_state.prediction is not None:
            _, predicted_digit, confidence = st.session_state.prediction

            st.caption("THE CNN PREDICTS")
            st.markdown(
                f"<div style='font-size:88px; font-weight:750; "
                f"line-height:1.1; text-align:center;'>{predicted_digit}</div>",
                unsafe_allow_html=True
            )
            st.divider()
            st.metric("Confidence", f"{confidence * 100:.2f}%")
            st.progress(confidence)

            st.markdown(
                "Built with MNIST's **60,000** handwritten training digits "
                "(**6,000** reserved for validation) and evaluated on "
                "**10,000** test images from `tensorflow.keras.datasets.mnist`. "
                "Wrong answer? Your handwriting must be super… unique! "
                "Just kidding—the CNN needs more practice, not you. 😄 "
                "Share the correct digit and your doodle may join a future "
                "training iteration. No instant learning or guaranteed improvement!"
            )
            with st.container(key="wrong_feedback"):
                st.markdown(
                    "<style>.st-key-wrong_feedback button {"
                    "background-color:#b91c1c !important;color:white !important;"
                    "border-color:#b91c1c !important;font-weight:800 !important;"
                    "}</style>",
                    unsafe_allow_html=True
                )
                if st.button(
                    "Wrong?", key="wrong_button", use_container_width=True,
                    disabled=st.session_state.feedback_saved
                ):
                    st.session_state.show_feedback = True

            if st.session_state.feedback_saved:
                st.success("Saved! Your digit has joined the CNN's homework pile. 📝")
            elif st.session_state.show_feedback:
                st.caption(
                    "With your permission, we save the processed 28×28 PNG, "
                    "predicted/correct digits, timestamp and a submission ID "
                    "in private Google Drive/Sheets for review and future training. "
                    "No name or original canvas is collected."
                )
                with st.form(f"correction_{st.session_state.feedback_id}"):
                    correct_digit = st.selectbox(
                        "What digit did you mean?",
                        [digit for digit in range(10) if digit != predicted_digit],
                        index=None, placeholder="Choose the correct digit"
                    )
                    consent = st.checkbox("I agree to save this digit for future training.")
                    submitted = st.form_submit_button("Submit correction")
                if submitted:
                    if correct_digit is None or not consent:
                        st.warning("Choose a digit and agree to save it first.")
                    else:
                        try:
                            feedback_image = st.session_state.normalized_image
                            feedback_id = st.session_state.feedback_id
                            if feedback_image is None or feedback_id is None:
                                raise ValueError("Predict the current drawing first.")
                            with st.spinner("Sending homework to the CNN…"):
                                submit_feedback(
                                    feedback_image,
                                    predicted_digit, correct_digit,
                                    feedback_id
                                )
                            st.session_state.feedback_saved = True
                            st.session_state.show_feedback = False
                            st.rerun()
                        except Exception as error:
                            show_feedback_failure(error)
                            st.caption("Retrying reuses this submission ID; a saved correction will not be duplicated.")
        else:
            st.markdown(
                "<div style='font-size:76px; font-weight:700; "
                "line-height:1.2; text-align:center; opacity:0.25;'>?</div>",
                unsafe_allow_html=True
            )
            st.caption("Your predicted digit and confidence will appear here.")

if predict_clicked and drawing_available:
    try:
        with col_result:
            with st.container(border=True):
                with st.spinner("The CNN is analysing your digit..."):
                    run_full_pipeline(image_data)
        st.rerun()
    except Exception as error:
        st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# 7. PIPELINE EXECUTION
# --------------------------------------------------

st.divider()
st.subheader("Step 2 — Explore the processing pipeline")
st.write(
    "You can execute each stage individually to inspect how the image "
    "is prepared before the CNN makes its prediction."
)


# --------------------------------------------------
# 8. INDIVIDUAL PROCESSING STEPS
# --------------------------------------------------

step1, step2, step3, step4 = st.columns(4)

with step1:
    if st.button(
        "1. Grayscale",
        use_container_width=True,
        disabled=not drawing_available
    ):
        reset_pipeline()
        st.session_state.gray_image = convert_to_grayscale(image_data)
        st.session_state.resized_image = None
        st.session_state.normalized_image = None
        st.session_state.prediction = None

with step2:
    if st.button(
        "2. Resize & centre",
        use_container_width=True,
        disabled=st.session_state.gray_image is None
    ):
        st.session_state.show_feedback = False
        st.session_state.feedback_saved = False
        st.session_state.feedback_id = None
        st.session_state.resized_image = resize_image(
            st.session_state.gray_image
        )
        st.session_state.normalized_image = None
        st.session_state.prediction = None

with step3:
    if st.button(
        "3. Normalize",
        use_container_width=True,
        disabled=st.session_state.resized_image is None
    ):
        st.session_state.show_feedback = False
        st.session_state.feedback_saved = False
        st.session_state.feedback_id = None
        normalized, _ = normalize_image(st.session_state.resized_image)
        st.session_state.normalized_image = normalized
        st.session_state.prediction = None

with step4:
    if st.button(
        "4. Predict",
        use_container_width=True,
        disabled=st.session_state.normalized_image is None
    ):
        _, model_input = normalize_image(st.session_state.resized_image)
        st.session_state.prediction = predict_digit(model_input)
        st.session_state.show_feedback = False
        st.session_state.feedback_saved = False
        st.session_state.feedback_id = str(uuid.uuid4())
        st.rerun()


# --------------------------------------------------
# 9. DISPLAY PREPROCESSING VISUALLY
# --------------------------------------------------

st.divider()
st.subheader("Step 3 — What the CNN actually sees")
st.caption(
    "Follow the image from the original drawing to the final normalized "
    "28 × 28 input."
)

image_col1, image_col2, image_col3 = st.columns(3)

with image_col1:
    st.markdown("**① Grayscale image**")
    if st.session_state.gray_image is not None:
        st.image(
            st.session_state.gray_image,
            clamp=True,
            use_container_width=True
        )
        st.caption("One brightness value per pixel.")
    else:
        st.info("Waiting for grayscale processing.")

with image_col2:
    st.markdown("**② Cropped, resized and centred**")
    if st.session_state.resized_image is not None:
        st.image(
            st.session_state.resized_image.resize(
                (280, 280),
                Image.Resampling.NEAREST
            ),
            clamp=True,
            use_container_width=True
        )
        st.caption("28 × 28 pixels, enlarged here so you can inspect it.")
    else:
        st.info("Waiting for resizing and centring.")

with image_col3:
    st.markdown("**③ Normalized input**")
    if st.session_state.normalized_image is not None:
        st.image(
            st.session_state.normalized_image,
            clamp=True,
            use_container_width=True
        )
        st.caption("Pixel values are now between 0 and 1.")
    else:
        st.info("Waiting for normalization.")


# --------------------------------------------------
# 10. MODEL PREDICTION
# --------------------------------------------------

st.divider()
st.subheader("Step 4 — CNN prediction")

if st.session_state.prediction is not None:
    probabilities, predicted_digit, confidence = st.session_state.prediction

    result_col, confidence_col = st.columns(2)

    with result_col:
        st.metric("Predicted digit", predicted_digit)

    with confidence_col:
        st.metric("Prediction confidence", f"{confidence * 100:.2f}%")

    st.markdown("### Probability distribution")
    st.caption(
        "The CNN assigns a probability to each possible digit. "
        "The highest probability becomes the prediction."
    )

    probability_data = {
        "Digit": [str(i) for i in range(10)],
        "Probability": probabilities
    }

    st.bar_chart(
        probability_data,
        x="Digit",
        y="Probability",
        horizontal=False,
        height=350
    )

    st.markdown("### Individual confidence levels")
    confidence_cols = st.columns(5)

    for digit in range(10):
        with confidence_cols[digit % 5]:
            st.write(f"**Digit {digit}**")
            st.progress(float(probabilities[digit]))
            st.caption(f"{probabilities[digit] * 100:.2f}%")

    st.success(
        f"The CNN predicts **{predicted_digit}** "
        f"with {confidence * 100:.2f}% confidence."
    )
else:
    st.info(
        "Draw a digit and select **Predict digit** beside the canvas "
        "to see the result."
    )


# --------------------------------------------------
# 11. MODEL INFORMATION
# --------------------------------------------------

with st.expander("About this CNN model"):
    st.write("Architecture: Convolutional Neural Network")
    st.write("Dataset: MNIST handwritten digits")
    st.write("Input shape: 28 × 28 × 1")
    st.write("Output classes: 10 (digits 0–9)")
    st.write("Output activation: Softmax")
    st.write(
        "The model returns a probability distribution "
        "across all ten digit classes."
    )
