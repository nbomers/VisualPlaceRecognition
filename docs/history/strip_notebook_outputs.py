# Notebook-Ausgaben aus jeder Version entfernen -- die Versionen selbst bleiben.
import json
if filename.endswith(b".ipynb"):
    roh = value.get_contents_by_identifier(blob_id)
    try:
        nb = json.loads(roh.decode("utf-8"))
    except Exception:
        return (filename, mode, blob_id)
    geaendert = False
    for zelle in nb.get("cells", []):
        if zelle.get("cell_type") == "code":
            if zelle.get("outputs") or zelle.get("execution_count") is not None:
                zelle["outputs"] = []
                zelle["execution_count"] = None
                geaendert = True
        if "metadata" in zelle and "execution" in zelle["metadata"]:
            del zelle["metadata"]["execution"]
            geaendert = True
    if geaendert:
        neu = json.dumps(nb, indent=1, ensure_ascii=False).encode("utf-8") + b"\n"
        blob_id = value.insert_file_with_contents(neu)
return (filename, mode, blob_id)
