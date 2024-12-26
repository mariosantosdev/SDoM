# Stable Diffusion (A1111)
#
# This example runs the popular https://github.com/AUTOMATIC1111/stable-diffusion-webui
#
# To run a temporary A1111 server execute `modal serve a1111_webui.py`
# To deploy it permanently execute `modal deploy a1111_webui.py`

from collections.abc import Mapping
import subprocess

import modal
import os

PORT = 8000
MODELS_PATH = "/webui/models/Stable-diffusion"
SOURCES = ["hugging_face", "civit"]

# First, we define the image A1111 will run in.
a1111_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "wget",
        "git",
        "libgl1",
        "libglib2.0-0",
        "google-perftools",  # For tcmalloc
    )
    .pip_install(
        "huggingface-hub[hf_transfer]==0.25.2",
        "civitdl"
        )
    .env({
        "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"
        })
    .run_commands(
        "git clone --depth 1 --branch v1.7.0 https://github.com/AUTOMATIC1111/stable-diffusion-webui /webui",
        "python -m venv /webui/venv",
        "cd /webui && . venv/bin/activate && "
        + "python -c 'from modules import launch_utils; launch_utils.prepare_environment()' --xformers",
        gpu="a10g",
    )
    .run_commands(
        "cd /webui && . venv/bin/activate && "
        + "python -c 'from modules import shared_init, initialize; shared_init.initialize(); initialize.initialize()'",
        gpu="a10g",
    )
    .run_commands(  # needs to be empty for Volume mount to work
        f"rm -rf {MODELS_PATH}",
        f"mkdir {MODELS_PATH}"
    )
)

app = modal.App("SDoM-a1111", image=a1111_image)
vol = modal.Volume.from_name("SDoM-sd-models", create_if_missing=True)
vol_settings = modal.Volume.from_name("SDoM-settings", create_if_missing=True)
secret = modal.Secret.from_dotenv(filename=".env")


@app.function(
    volumes={"/cache": vol_settings}
)
def read_models_from_file(model_source: str):
    import json

    if model_source.lower() not in SOURCES:
        raise modal.exception.ExecutionError("model source is invalid.")

    with vol_settings.batch_upload(force=True) as batch:
        batch.put_file("./models.json", "/models.json")

    data = b""
    for chunk in vol_settings.read_file("models.json"):
        data += chunk
    json_file = json.loads(data)

    return json_file[model_source]


@app.function(
    volumes={MODELS_PATH: vol},
)
def hf_download(model: Mapping[str, str]):
    from huggingface_hub import hf_hub_download

    dir_name = model["repo"].split("/")[-1]

    hf_hub_download(
        repo_id=model["repo"],
        filename=model["filename"],
        local_dir=os.path.join(MODELS_PATH, dir_name),
    )
    vol.commit()


@app.function(
    volumes={MODELS_PATH: vol},
    secrets=[secret]
)
def civit_download():
    from civitdl.batch.batch_download import batch_download, BatchOptions

    model_ids = read_models_from_file.remote(model_source="civit")

    batchOptions = BatchOptions(
        sorter="basic",
        max_images=None,
        nsfw_mode='2',
        api_key=os.getenv("CIVIT_API"),

        with_prompt=None,
        without_model=None,
        limit_rate=None,
        retry_count=None,
        pause_time=None,

        cache_mode=None,
        strict_mode=None,
        model_overwrite=None,

        with_color=None,
        verbose=None
    )

    batch_download(
        source_strings=list(model_ids),
        rootdir=MODELS_PATH,
        batchOptions=batchOptions
    )
    vol.commit()


@app.local_entrypoint()
def download_models():
    civit_download.remote()

    hf_models = read_models_from_file.remote(model_source="hugging_face")
    list(hf_download.map(hf_models))


# If you want to run it with an A100 or H100 GPU, just change `gpu="a10g"` to `gpu="a100"` or `gpu="h100"`.
#
# Startup of the web server should finish in under one to three minutes.
@app.function(
    gpu="a10g",
    cpu=2,
    memory=1024,
    timeout=3600,
    # Allows 100 concurrent requests per container.
    allow_concurrent_inputs=100,
    # Keep at least one instance of the server running.
    keep_warm=1,
    volumes={MODELS_PATH: vol}
)
@modal.web_server(port=PORT, startup_timeout=180)
def run():
    START_COMMAND = f"""
cd /webui && \
. venv/bin/activate && \
accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision=fp16 \
    --dynamo_backend=inductor \
    --num_cpu_threads_per_process=6 \
    /webui/launch.py \
        --skip-prepare-environment \
        --no-gradio-queue \
        --listen \
        --port {PORT}
"""
    subprocess.Popen(START_COMMAND, shell=True)
