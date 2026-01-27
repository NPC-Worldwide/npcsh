import os
import glob
import yaml
import pytest


def get_jinx_files():
    """Get all jinx files in the npc_team directory."""
    base_path = os.path.join(os.path.dirname(__file__), '..', 'npcsh', 'npc_team', 'jinxs')
    return glob.glob(os.path.join(base_path, '**', '*.jinx'), recursive=True)


@pytest.mark.parametrize("jinx_file", get_jinx_files())
def test_jinx_yaml_valid(jinx_file):
    """Test that each jinx file has valid YAML syntax (or valid Jinja template)."""
    with open(jinx_file) as f:
        content = f.read()

    # Skip files with Jinja control flow - they're processed before YAML parsing
    if '{%' in content:
        pytest.skip("Jinx contains Jinja templates - requires preprocessing")

    data = yaml.safe_load(content)

    assert data is not None, f"Jinx file {jinx_file} is empty"
    assert 'jinx_name' in data, f"Jinx file {jinx_file} missing jinx_name"


@pytest.mark.parametrize("jinx_file", get_jinx_files())
def test_jinx_has_required_fields(jinx_file):
    """Test that each jinx file has required fields."""
    with open(jinx_file) as f:
        content = f.read()

    # Skip files with Jinja control flow
    if '{%' in content:
        pytest.skip("Jinx contains Jinja templates - requires preprocessing")

    data = yaml.safe_load(content)

    assert 'jinx_name' in data
    assert 'steps' in data or 'description' in data


def test_search_jinxs_exist():
    """Test that the search jinxs exist."""
    base_path = os.path.join(os.path.dirname(__file__), '..', 'npcsh', 'npc_team', 'jinxs', 'lib', 'core')

    # Main search jinx
    assert os.path.exists(os.path.join(base_path, 'search.jinx'))

    # Sub-search jinxs
    search_dir = os.path.join(base_path, 'search')
    assert os.path.exists(os.path.join(search_dir, 'web_search.jinx'))
    assert os.path.exists(os.path.join(search_dir, 'mem_search.jinx'))
    assert os.path.exists(os.path.join(search_dir, 'kg_search.jinx'))
    assert os.path.exists(os.path.join(search_dir, 'file_search.jinx'))
    assert os.path.exists(os.path.join(search_dir, 'db_search.jinx'))
