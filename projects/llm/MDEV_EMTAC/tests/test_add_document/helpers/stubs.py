class DummyResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {"image_id": 123}

    def json(self):
        return self._payload


def stub_requests_post_success(*args, **kwargs):
    return DummyResponse(status_code=200, payload={"image_id": 123})


def stub_requests_post_fail(*args, **kwargs):
    return DummyResponse(status_code=500, payload={"error": "forced failure"})


def stub_generate_embedding(text, model):
    # deterministic embedding
    return [0.0] * 10
