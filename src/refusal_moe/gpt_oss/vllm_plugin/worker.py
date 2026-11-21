"""Worker extension exposing the routing controls as named, data-only RPCs."""

from . import controls


class ResearchWorker:
    def research_control(self, operation, options=None):
        functions = {
            "describe": controls.describe,
            "install": controls.install,
            "set_condition": controls.set_condition,
            "collect": controls.collect,
        }
        if operation not in functions:
            raise ValueError(operation)
        return functions[operation](self.model_runner.model, **(options or {}))
