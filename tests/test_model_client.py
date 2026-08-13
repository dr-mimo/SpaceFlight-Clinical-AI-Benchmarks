import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model_client import ModelClient  # noqa: E402


def test_mock_backend_requires_no_network_or_credentials():
    # Should not raise even with no GOOGLE_CLOUD_PROJECT / edge server present.
    client = ModelClient(backend="mock")
    resp = client.query("mock-model-a", "A crewmember presents with suspected decompression sickness.")
    assert resp.backend == "mock"
    assert resp.error is None
    assert "decompression" in resp.text.lower() or "mock" in resp.text.lower()


def test_mock_backend_is_deterministic_for_same_input():
    client = ModelClient(backend="mock")
    prompt = "Suspected blast overpressure injury with chest pain."
    r1 = client.query("mock-model-a", prompt)
    r2 = client.query("mock-model-a", prompt)
    assert r1.text == r2.text


def test_mock_backend_varies_by_model_and_prompt():
    client = ModelClient(backend="mock")
    r_a = client.query("mock-model-a", "cardiac arrhythmia in microgravity")
    r_b = client.query("mock-model-b", "cardiac arrhythmia in microgravity")
    # Different model_id changes the embedded hash even if scenario text matches.
    assert r_a.text != r_b.text


def test_invalid_backend_raises():
    try:
        ModelClient(backend="not-a-real-backend")  # type: ignore[arg-type]
        assert False, "expected ValueError for unknown backend"
    except ValueError:
        pass


def test_vertex_backend_requires_project():
    # No GOOGLE_CLOUD_PROJECT set in test environment and none passed explicitly.
    env_backup = os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
    try:
        try:
            ModelClient(backend="vertex")
            assert False, "expected ValueError when no project is configured"
        except ValueError:
            pass
    finally:
        if env_backup is not None:
            os.environ["GOOGLE_CLOUD_PROJECT"] = env_backup
