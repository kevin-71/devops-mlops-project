from backend import train as train_module
from backend.climate.service import ClimateService


def test_run_training_entrypoint_uses_refresh_flag(monkeypatch):
    calls: dict[str, object] = {}

    class DummyService:
        def train(self, force: bool = False):
            calls["force"] = force
            return {"summary": {}, "metrics": {}}

    monkeypatch.setattr(train_module, "ClimateService", lambda: DummyService())

    result = train_module.run_training(refresh=True)

    assert result == {"summary": {}, "metrics": {}}
    assert calls["force"] is True


def test_service_reset_state_clears_in_memory_cache():
    service = ClimateService()
    service._state = {"summary": {"rows": 1}}

    service.reset_state()

    assert service._state is None
