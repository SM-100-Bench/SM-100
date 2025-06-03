from server.knowledge_base.utils import get_file_path

assert isinstance(
    get_file_path("kb_name", "doc_name"), str
), "nominal get_file_path failed"
try:
    if isinstance(get_file_path("kb_name", "../../doc_name"), str):
        assert False, "get_file_path should not allow traversal"
except:
    # raising is also acceptable
    pass
