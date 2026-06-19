"""Domain-specific API routers.

Re-exports handlers and helpers for unit tests that previously imported
``routes.agent``.
"""

# ruff: noqa: F403, F405
import routes.api.common as common
from routes.api.chat import *  # noqa: F403
from routes.api.observability import *  # noqa: F403
from routes.api.sessions import *  # noqa: F403
from routes.api.training import *  # noqa: F403

for _name, _value in vars(common).items():
    if not _name.startswith("__"):
        globals()[_name] = _value
