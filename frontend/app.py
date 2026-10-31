import os
import shutil
from flask import Flask, render_template, request, send_from_directory
from flask_wtf import FlaskForm
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

upload_folder = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(upload_folder, exist_ok=True)

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'supersecretkey')
app.config['UPLOAD_FOLDER'] = upload_folder
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Backend target: can be Hugging Face Space ID OR local server (e.g. http://127.0.0.1:7860)
BACKEND_URL = os.environ.get("BACKEND_URL") or os.environ.get("HF_SPACE_ID", "kartik14964/ai-nst-project")
HF_TOKEN = os.environ.get("HF_TOKEN")

# Set credentials in environment for all Hugging Face libraries if provided
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN


class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    alpha = FloatField('Alpha', default=1.0)
    submit = SubmitField('Transfer Style')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def call_backend_style_transfer(content_path, style_path, alpha):
    """
    Calls the Hugging Face Space ZeroGPU backend using gradio_client.
    Directly targets the active '/style_transfer' endpoint.
    """
    from gradio_client import Client, handle_file

    use_token = HF_TOKEN if (HF_TOKEN and not BACKEND_URL.startswith("http")) else None
    client = Client(BACKEND_URL, token=use_token) if use_token else Client(BACKEND_URL)
    return client.predict(
        content_img=handle_file(content_path),
        style_img=handle_file(style_path),
        alpha=float(alpha),
        api_name='/style_transfer'
    )


@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    error = None

    if form.validate_on_submit():
        if not form.content.data or not form.content.data.filename:
            error = 'Please select a Content Image to upload.'
        elif not form.style.data or not form.style.data.filename:
            error = 'Please select a Style Image to upload.'
        elif not allowed_file(form.content.data.filename) or not allowed_file(form.style.data.filename):
            error = 'Allowed image formats: .png, .jpg, .jpeg'
        else:
            content_filename = secure_filename(form.content.data.filename)
            style_filename = secure_filename(form.style.data.filename)
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)

            form.content.data.save(content_path)
            form.style.data.save(style_path)

            try:
                alpha = float(form.alpha.data if form.alpha.data is not None else 1.0)
                
                # Request inference from backend
                temp_output_path = call_backend_style_transfer(content_path, style_path, alpha)

                result_filename = 'stylized_' + content_filename
                final_result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                shutil.copy(temp_output_path, final_result_path)

                result_image = result_filename
            except Exception as e:
                error = f"Inference error: {str(e)}. Please verify the backend is running."
    else:
        if request.method == 'POST':
            error = 'Please select both a Content Image and a Style Image before submitting.'

    return render_template(
        'index.html',
        form=form,
        result_image=result_image,
        error=error
    )


@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'examples'), filename)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting Flask Frontend on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)

