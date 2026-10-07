from http.client import IncompleteRead

import pytest

from unitwave.data.atlas_meshes import MESH_URL, mesh_path

OBJ = b"# test mesh\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"


def test_a_mesh_is_fetched_once_then_served_from_the_cache(tmp_path):
    calls = []

    def fetch(url):
        calls.append(url)
        return OBJ

    first = mesh_path(382, tmp_path, fetch=fetch)
    second = mesh_path(382, tmp_path, fetch=fetch)
    assert first == second == tmp_path / "ccf_2017_meshes" / "382.obj"
    assert first.read_bytes() == OBJ
    assert calls == [MESH_URL.format(382)]
    assert not list(first.parent.glob("*.tmp"))


def test_refuses_content_that_is_not_an_obj_mesh(tmp_path):
    with pytest.raises(ValueError, match="not an OBJ mesh"):
        mesh_path(382, tmp_path, fetch=lambda url: b"<html>Not found</html>")
    assert not (tmp_path / "ccf_2017_meshes" / "382.obj").exists()


@pytest.mark.parametrize("bad", [-1, 1.5, "382", True])
def test_refuses_ids_that_are_not_structure_ids(tmp_path, bad):
    with pytest.raises(ValueError, match="structure id"):
        mesh_path(bad, tmp_path, fetch=lambda url: OBJ)


def test_a_stalled_download_fails_in_plain_words_and_is_tried_again(tmp_path):
    # Real case (S1, session 6a601cc5): the download of structure 679 (CS) stalled
    # until the read timed out, and the page showed only "responded with 400".
    def stalled(url):
        raise TimeoutError("The read operation timed out")

    with pytest.raises(OSError, match=r"could not download the mesh for structure 679 .*timed out"):
        mesh_path(679, tmp_path, fetch=stalled)
    assert not (tmp_path / "ccf_2017_meshes").exists() or not list(
        (tmp_path / "ccf_2017_meshes").iterdir()
    )
    # Nothing was cached, so the next request downloads it.
    assert mesh_path(679, tmp_path, fetch=lambda url: OBJ).read_bytes() == OBJ


def test_a_download_cut_short_fails_in_plain_words(tmp_path):
    # Real case (S1, session 6a601cc5, Allen level): structure 771 (P) stopped after
    # 63508 of 916133 bytes. IncompleteRead is not an OSError, so the server's refusal
    # missed it and the page got an empty response.
    def cut(url):
        raise IncompleteRead(b"v" * 63508, 852625)

    with pytest.raises(OSError, match=r"could not download the mesh for structure 771 "):
        mesh_path(771, tmp_path, fetch=cut)
    assert not (tmp_path / "ccf_2017_meshes" / "771.obj").exists()


def test_one_failed_download_is_retried_once(tmp_path):
    calls = []

    def flaky(url):
        calls.append(url)
        if len(calls) == 1:
            raise TimeoutError("The read operation timed out")
        return OBJ

    assert mesh_path(679, tmp_path, fetch=flaky).read_bytes() == OBJ
    assert calls == [MESH_URL.format(679)] * 2
