from pathlib import Path
import zipfile

try:
    from google.colab import files
except ImportError:
    files = None


ROOT = Path.cwd()
ARCHIVE = ROOT / "assignment3_artifacts.zip"

ARTIFACTS = [
    "model.bin",
    # "model.bin.optim",
    "vocab.json",
    "outputs",
    "runs",
    "logs",
]


def add_path(archive, path):
    if path.is_file():
        archive.write(path, arcname=path.relative_to(ROOT))
    elif path.is_dir():
        for child in path.rglob("*"):
            if child.is_file():
                archive.write(child, arcname=child.relative_to(ROOT))


with zipfile.ZipFile(
    ARCHIVE,
    "w",
    compression=zipfile.ZIP_DEFLATED,
) as archive:
    for relative_path in ARTIFACTS:
        path = ROOT / relative_path
        if path.exists():
            add_path(archive, path)

size_mb = ARCHIVE.stat().st_size / 1024**2
print(f"Created {ARCHIVE.name} ({size_mb:.2f} MB)")

if files is not None:
    files.download(str(ARCHIVE))
else:
    print("google.colab.files is unavailable; download the archive manually.")