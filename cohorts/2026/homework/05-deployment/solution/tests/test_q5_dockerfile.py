from q5_dockerfile_base import base_image


def test_base_image_tag():
    base = base_image()
    assert f"{base['image']}:{base['tag']}" == "python:3.11.15-slim-bookworm"
    assert base["digest"].startswith("sha256:")
