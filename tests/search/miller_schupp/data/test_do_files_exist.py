import rlformath.search.miller_schupp.data as data
from importlib import resources

def test_do_miller_schupp_txt_files_exist():
    for file_type in ["greedy_solved", "all"]:
        file_name = f"{file_type}_presentations.txt"
        file_path = resources.files(data) / file_name
        assert file_path.is_file(), f"File {file_name} does not exist in the package"