import os
import shutil
from flask import Flask, render_template, request, send_from_directory
from flask_wtf import FlaskForm
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField

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

# Backend Hugging Face Space configuration
HF_SPACE_ID = os.environ.get("HF_SPACE_ID", "kartik14964/ai-nst-project")
HF_TOKEN = os.environ.get("HF_TOKEN")

# Automatically set credentials if provided via environment
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN


class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    content_path = HiddenField()
    style_path = HiddenField()
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

    client = Client(HF_SPACE_ID)
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
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():
        if form.content.data and form.content.data.filename:
            if allowed_file(form.content.data.filename):
                content_filename = secure_filename(form.content.data.filename)
                form.content.data.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))
                form.content_path.data = content_filename
        else:
            content_filename = form.content_path.data

        if form.style.data and form.style.data.filename:
            if allowed_file(form.style.data.filename):
                style_filename = secure_filename(form.style.data.filename)
                form.style.data.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
                form.style_path.data = style_filename
        else:
            style_filename = form.style_path.data

        # Validate that both files exist on disk before showing any preview or calling backend
        content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename) if content_filename else None
        style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename) if style_filename else None

        if not content_path or not os.path.isfile(content_path):
            error = 'Please select a Content Image to upload.'
            content_filename = None
            form.content_path.data = ''
        elif not style_path or not os.path.isfile(style_path):
            error = 'Please select a Style Image to upload.'
            style_filename = None
            form.style_path.data = ''
        else:
            try:
                alpha = float(form.alpha.data if form.alpha.data is not None else 1.0)
                
                # Request inference from HF ZeroGPU backend
                temp_output_path = call_backend_style_transfer(content_path, style_path, alpha)

                result_filename = 'stylized_' + content_filename
                final_result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                shutil.copy(temp_output_path, final_result_path)

                result_image = result_filename
            except Exception as e:
                error = f"Inference error: {str(e)}. Please verify the Hugging Face Space backend is running."
    else:
        if request.method == 'POST':
            error = 'Please select both a Content Image and a Style Image before submitting.'
            content_filename = None
            style_filename = None
            form.content_path.data = ''
            form.style_path.data = ''

    return render_template(
        'index.html',
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
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

