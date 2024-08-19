import rlformath.search.miller_schupp.data as data
from importlib import resources

def test_do_miller_schupp_txt_files_exist():
    for states_type in ["solved", "all"]:
        file_name = f"{states_type}_miller_schupp_presentations.txt"
        file_path = resources.files(data) / file_name
        assert file_path.is_file(), f"File {file_name} does not exist in the package"