"""
Smoke test: verify that Stage 0-1 dependencies are importable.
No real logic is tested here yet — this just confirms the environment is correct.
"""


def test_pillow_importable():
    import PIL.Image  # noqa: F401


def test_pillow_heif_importable():
    import pillow_heif  # noqa: F401


def test_imagehash_importable():
    import imagehash  # noqa: F401
