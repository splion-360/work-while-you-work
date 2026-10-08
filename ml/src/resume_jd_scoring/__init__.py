__all__ = ["audit_rows", "load_dataset_rows"]


def __getattr__(name):
    if name in __all__:
        from resume_jd_scoring.data import audit_rows, load_dataset_rows

        return {"audit_rows": audit_rows, "load_dataset_rows": load_dataset_rows}[name]
    raise AttributeError(name)
