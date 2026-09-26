from src.normalize import name_views, addr_views


def test_legal_suffix_variants_collapse():
    a, b = name_views("ABC Trading Pvt. Ltd."), name_views("abc trading private limited")
    assert a["name_core"] == b["name_core"] == "abc trading"
    assert a["name_legal"] == b["name_legal"]


def test_city_tokens_survive():
    assert name_views("ABC Trading Mumbai")["name_core"] != name_views("ABC Trading Delhi")["name_core"]


def test_accents_and_ampersand():
    assert name_views("Café & Crème SARL")["name_core"] == "cafe and creme"


def test_address_views():
    v = addr_views("12, M.G. Rd, Near SBI ATM, Pune 411001")
    assert "road" in v["addr_norm"].split()
    assert v["addr_postal"] == "411001" and v["addr_first_num"] == "12" and v["addr_landmark"] == 1
