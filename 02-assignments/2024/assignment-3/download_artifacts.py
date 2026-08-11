from __future__ import annotations

import argparse
import os
import zipfile
from getpass import getpass
from pathlib import Path
from urllib.parse import quote

import requests

# Usage
# For the release you already created:

# !python download_artifacts.py \
#     --repository ilyarudiak/CS224N-NLP-with-DL-2024 \
#     --tag colab-artifacts-2026-08-11 \
#     --replace
    
# For a new release in the future:

# !python download_artifacts.py \
#     --repository ilyarudyak/CS224N-NLP-with-DL-2024 \
#     --tag colab-artifacts-2026-08-12 \
#     --create-release


DEFAULT_REPOSITORY = "ilyarudyak/CS224N-NLP-with-DL-2024"
DEFAULT_ARCHIVE = "assignment3_artifacts.zip"

ARTIFACTS = [
    "model.bin",
    # "model.bin.optim",
    "vocab.json",
    "outputs",
    "runs",
    "logs",
]

API_BASE_URL = "https://api.github.com"
API_VERSION = "2022-11-28"


def add_path_to_archive(
    archive: zipfile.ZipFile,
    path: Path,
    root: Path,
) -> None:
    """Add one file or directory to the archive."""
    if path.is_file():
        archive.write(path, arcname=path.relative_to(root))
        return

    if path.is_dir():
        for child in path.rglob("*"):
            if child.is_file():
                archive.write(child, arcname=child.relative_to(root))


def create_archive(
    root: Path,
    archive_path: Path,
) -> Path:
    """Create the ZIP archive from the configured artifact paths."""
    required_artifacts = {"model.bin", "vocab.json"}

    missing_required = [
        relative_path
        for relative_path in required_artifacts
        if not (root / relative_path).exists()
    ]

    if missing_required:
        raise FileNotFoundError(
            "Required artifacts are missing: "
            + ", ".join(sorted(missing_required))
        )

    missing_optional = []

    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for relative_path in ARTIFACTS:
            path = root / relative_path

            if path.exists():
                add_path_to_archive(archive, path, root)
            else:
                missing_optional.append(relative_path)

    if missing_optional:
        print(
            "Skipped missing optional paths: "
            + ", ".join(missing_optional)
        )

    size_mb = archive_path.stat().st_size / 1024**2
    print(f"Created {archive_path} ({size_mb:.2f} MB)")

    return archive_path


def github_request(
    method: str,
    url: str,
    token: str,
    *,
    timeout: int = 60,
    **kwargs,
):
    """Make an authenticated GitHub API request."""
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": API_VERSION,
    }
    headers.update(kwargs.pop("headers", {}))

    response = requests.request(
        method,
        url,
        headers=headers,
        timeout=timeout,
        **kwargs,
    )

    if not response.ok:
        try:
            details = response.json()
        except ValueError:
            details = response.text

        raise RuntimeError(
            f"GitHub API request failed ({response.status_code}): "
            f"{details}"
        )

    if not response.content:
        return None

    return response.json()


def get_or_create_release(
    repository: str,
    tag: str,
    token: str,
    *,
    create_if_missing: bool,
) -> dict:
    """Return a release identified by tag, optionally creating it."""
    encoded_tag = quote(tag, safe="")
    release_url = (
        f"{API_BASE_URL}/repos/{repository}/releases/tags/{encoded_tag}"
    )

    response = requests.get(
        release_url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": API_VERSION,
        },
        timeout=60,
    )

    if response.ok:
        return response.json()

    if response.status_code != 404 or not create_if_missing:
        try:
            details = response.json()
        except ValueError:
            details = response.text

        raise RuntimeError(
            f"Could not find release {tag!r} "
            f"({response.status_code}): {details}"
        )

    print(f"Release {tag!r} does not exist; creating it.")

    return github_request(
        "POST",
        f"{API_BASE_URL}/repos/{repository}/releases",
        token,
        json={
            "tag_name": tag,
            "name": tag,
            "body": "Artifacts uploaded from a Colab runtime.",
            "draft": False,
            "prerelease": False,
        },
    )


def delete_existing_asset(
    repository: str,
    asset: dict,
    token: str,
) -> None:
    """Delete an existing release asset."""
    asset_url = (
        f"{API_BASE_URL}/repos/{repository}/releases/assets/"
        f"{asset['id']}"
    )

    github_request(
        "DELETE",
        asset_url,
        token,
    )

    print(f"Deleted existing asset: {asset['name']}")


def upload_asset(
    repository: str,
    release: dict,
    archive_path: Path,
    token: str,
    *,
    replace_existing: bool,
) -> dict:
    """Upload the archive to the specified GitHub Release."""
    asset_name = archive_path.name

    existing_asset = next(
        (
            asset
            for asset in release.get("assets", [])
            if asset["name"] == asset_name
        ),
        None,
    )

    if existing_asset is not None:
        if not replace_existing:
            raise RuntimeError(
                f"The release already contains {asset_name!r}. "
                "Use --replace to delete and upload it again."
            )

        delete_existing_asset(repository, existing_asset, token)

    upload_url = release["upload_url"].split("{", 1)[0]

    print(f"Uploading {archive_path} to GitHub...")

    with archive_path.open("rb") as archive_file:
        uploaded_asset = github_request(
            "POST",
            upload_url,
            token,
            params={"name": asset_name},
            headers={"Content-Type": "application/zip"},
            data=archive_file,
            timeout=3600,
        )

    return uploaded_asset


def get_github_token() -> str:
    """Get the token without requiring it to be written in the notebook."""
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token

    # Optional support for Colab Secrets.
    try:
        from google.colab import userdata

        token = userdata.get("GITHUB_TOKEN")
        if token:
            return token
    except (ImportError, KeyError):
        pass

    token = getpass("GitHub token: ")

    if not token:
        raise RuntimeError("No GitHub token was provided.")

    return token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and upload assignment artifacts to GitHub."
    )

    parser.add_argument(
        "--repository",
        default=DEFAULT_REPOSITORY,
        help=f"GitHub repository, default: {DEFAULT_REPOSITORY}",
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="GitHub Release tag, for example colab-artifacts-2026-08-11",
    )
    parser.add_argument(
        "--archive",
        default=DEFAULT_ARCHIVE,
        help=f"Archive filename, default: {DEFAULT_ARCHIVE}",
    )
    parser.add_argument(
        "--create-release",
        action="store_true",
        help="Create the release if the tag does not exist.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace an existing asset with the same filename.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path.cwd()
    archive_path = Path(args.archive)

    if not archive_path.is_absolute():
        archive_path = root / archive_path

    token = get_github_token()

    create_archive(
        root=root,
        archive_path=archive_path,
    )

    release = get_or_create_release(
        repository=args.repository,
        tag=args.tag,
        token=token,
        create_if_missing=args.create_release,
    )

    asset = upload_asset(
        repository=args.repository,
        release=release,
        archive_path=archive_path,
        token=token,
        replace_existing=args.replace,
    )

    print("Uploaded successfully.")
    print(f"Release page: {release['html_url']}")
    print(f"Download URL: {asset['browser_download_url']}")


if __name__ == "__main__":
    main()