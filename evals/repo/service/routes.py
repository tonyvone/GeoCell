"""HTTP-style routing table and dispatch."""
ROUTES = {}


def register(method, path, handler):
    ROUTES[(method, path)] = handler


def normalize_user_path(path: str) -> str:
    return path.rstrip("/")


def normalize_order_path(path: str) -> str:
    return path.rstrip("/")


def normalize_report_path(path: str) -> str:
    return path.rstrip("/")


def normalize_event_path(path: str) -> str:
    return path.rstrip("/")


def normalize_invoice_path(path: str) -> str:
    return path.rstrip("/")


def list_users():
    return ["alice", "bob"]


def create_user():
    return "created-user"


def list_orders():
    return ["o-1", "o-2"]


def create_order():
    return "created-order"


def list_reports():
    return ["r-1"]


def create_report():
    return "created-report"


register("GET", "/users", list_users)
register("POST", "/users", create_user)
register("GET", "/orders", list_orders)
register("POST", "/orders", create_order)
register("GET", "/reports", list_reports)
register("POST", "/reports", create_report)


def dispatch(method: str, path: str):
    handler = ROUTES.get((method, path))
    if handler is None:
        raise LookupError(f"no route {method} {path}")
    return handler()
