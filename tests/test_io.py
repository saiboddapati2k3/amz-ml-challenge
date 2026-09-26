from src.io_utils import write_matching, read_id_list_file


def test_writer_enforces_rules(tmp_path):
    p = tmp_path / "m.tsv"
    write_matching(p, {"S1-1": ["S2-1", "S2-1", "S1-9", "S3-2"]}, ["S1-1", "S1-2", "S1-1"])
    lines = p.read_text().splitlines()
    assert lines == ["source1_entity_id\tmatched_entity_ids", "S1-1\tS2-1,S3-2", "S1-2\t"]
    assert read_id_list_file(p) == {"S1-1": ["S2-1", "S3-2"], "S1-2": []}
