from gqldepth import (
    Document,
    Field,
    depth_of,
    enforce,
    expand_fragments,
    parse_query,
    query_batch,
)


def _spread_count(value) -> int:
    if isinstance(value, Document):
        return sum(_spread_count(op.selections) for op in value.operations)
    if isinstance(value, Field):
        return sum(_spread_count(child) for child in value.selections)
    if isinstance(value, (list, tuple)):
        total = 0
        for child in value:
            if child.__class__.__name__ == "FragmentSpread":
                total += 1
            else:
                total += _spread_count(child)
        return total
    if value.__class__.__name__ == "InlineFragment":
        return _spread_count(value.selections)
    return 0


def test_parse_query_builds_nested_tree():
    ast = parse_query("query GetViewer { viewer { id name } }")

    assert isinstance(ast, Document)
    assert len(ast.operations) == 1
    operation = ast.operations[0]
    assert operation.operation == "query"
    assert operation.name == "GetViewer"
    assert len(operation.selections) == 1

    viewer = operation.selections[0]
    assert isinstance(viewer, Field)
    assert viewer.name == "viewer"
    assert [field.name for field in viewer.selections] == ["id", "name"]
    assert depth_of(ast) == 2


def test_depth_counts_nested_field_levels():
    ast = parse_query("{ viewer { profile { name } } }")

    assert depth_of(ast) == 3


def test_fragment_expansion_preserves_nested_depth():
    ast = parse_query(
        """
        query {
          viewer {
            ...ViewerFields
          }
        }
        fragment ViewerFields on Viewer {
          id
          profile {
            bio
          }
        }
        """
    )

    expanded = expand_fragments(ast, ast.fragments)
    assert _spread_count(expanded) == 0

    viewer = expanded.operations[0].selections[0]
    assert isinstance(viewer, Field)
    assert [field.name for field in viewer.selections] == ["id", "profile"]
    assert depth_of(expanded) == 3


def test_enforce_rejects_query_over_depth_limit():
    ast = parse_query("{ a { b { c } } }")
    policy = {
        "max_depth": 2,
        "max_fields": 100,
        "max_aliases": 100,
        "allow_introspection": False,
        "max_introspection_depth": 0,
    }

    errors = enforce(ast, policy)

    assert errors
    assert len(ast.operations[0].selections) == 1
    assert depth_of(ast) == 3


def test_query_batch_checks_each_query_independently():
    policy = {
        "max_depth": 2,
        "max_fields": 100,
        "max_aliases": 100,
        "allow_introspection": False,
        "max_introspection_depth": 0,
    }

    result = query_batch(["{ a { b } }", "{ a { b { c } } }"], policy)

    assert result["results"][0]["allowed"] is True
    assert result["results"][0]["errors"] == []
    assert result["results"][1]["allowed"] is False
    assert result["results"][1]["errors"]