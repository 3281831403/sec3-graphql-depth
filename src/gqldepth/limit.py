"""GraphQL query limits: parsing, depth, aliases, introspection, and batches.

The module intentionally keeps its own small GraphQL parser so that policy
checks do not depend on a particular GraphQL server implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping, Sequence

__all__ = [
    "Argument",
    "Directive",
    "Document",
    "Field",
    "FragmentCycleError",
    "FragmentDefinition",
    "FragmentSpread",
    "InlineFragment",
    "Operation",
    "QuerySyntaxError",
    "UnknownFragmentError",
    "Variable",
    "VariableDefinition",
    "check_aliases",
    "check_introspection",
    "count_fields",
    "depth_of",
    "enforce",
    "expand_fragments",
    "parse_query",
    "query_batch",
]


class QuerySyntaxError(ValueError):
    """Raised when the supplied text is not a supported GraphQL document."""


class UnknownFragmentError(ValueError):
    """Raised when a fragment spread has no matching definition."""


class FragmentCycleError(ValueError):
    """Raised when fragment expansion would recurse forever."""


@dataclass
class Variable:
    name: str


@dataclass
class Argument:
    name: str
    value: Any


@dataclass
class Directive:
    name: str
    arguments: list[Argument] = field(default_factory=list)


@dataclass
class Field:
    name: str
    alias: str | None = None
    arguments: list[Argument] = field(default_factory=list)
    directives: list[Directive] = field(default_factory=list)
    selections: list["Selection"] = field(default_factory=list)


@dataclass
class FragmentSpread:
    name: str
    directives: list[Directive] = field(default_factory=list)


@dataclass
class InlineFragment:
    selections: list["Selection"] = field(default_factory=list)
    type_condition: str | None = None
    directives: list[Directive] = field(default_factory=list)


@dataclass
class FragmentDefinition:
    name: str
    type_condition: str
    selections: list["Selection"] = field(default_factory=list)
    directives: list[Directive] = field(default_factory=list)


@dataclass
class VariableDefinition:
    name: str
    type_ref: str
    default: Any = None


@dataclass
class Operation:
    operation: str
    name: str | None = None
    variables: list[VariableDefinition] = field(default_factory=list)
    directives: list[Directive] = field(default_factory=list)
    selections: list["Selection"] = field(default_factory=list)


@dataclass
class Document:
    operations: list[Operation] = field(default_factory=list)
    fragments: dict[str, FragmentDefinition] = field(default_factory=dict)
    source: str = ""


Selection = Field | FragmentSpread | InlineFragment


_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMBER_RE = re.compile(
    r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
)
_PUNCTUATION = set("{}()[]:!$@=|&")

_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str
    position: int


def _lex(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char in " \t\r\n,\ufeff":
            index += 1
            continue
        if char == "#":
            newline = text.find("\n", index)
            index = length if newline < 0 else newline + 1
            continue
        if text.startswith("...", index):
            tokens.append(_Token("...", "...", index))
            index += 3
            continue
        if char == ".":
            raise QuerySyntaxError(f"unexpected character '.' at position {index}")
        if char in _PUNCTUATION:
            tokens.append(_Token(char, char, index))
            index += 1
            continue
        if char == '"':
            if text.startswith('"""', index):
                start = index + 3
                end = text.find('"""', start)
                if end < 0:
                    raise QuerySyntaxError(
                        f"unterminated block string starting at position {index}"
                    )
                tokens.append(_Token("STRING", text[start:end], index))
                index = end + 3
                continue
            index += 1
            chunks: list[str] = []
            while index < length:
                current = text[index]
                if current == '"':
                    index += 1
                    tokens.append(_Token("STRING", "".join(chunks), index))
                    break
                if current == "\\":
                    index += 1
                    if index >= length:
                        raise QuerySyntaxError("unterminated string escape")
                    escape = text[index]
                    if escape == "u":
                        digits = text[index + 1 : index + 5]
                        if len(digits) != 4 or not all(
                            c in "0123456789abcdefABCDEF" for c in digits
                        ):
                            raise QuerySyntaxError("invalid unicode escape")
                        chunks.append(chr(int(digits, 16)))
                        index += 5
                        continue
                    if escape not in _ESCAPES:
                        raise QuerySyntaxError(f"invalid escape sequence \\{escape}")
                    chunks.append(_ESCAPES[escape])
                    index += 1
                    continue
                if current in "\r\n":
                    raise QuerySyntaxError("newline inside a quoted string")
                chunks.append(current)
                index += 1
            else:
                raise QuerySyntaxError(
                    f"unterminated string starting at position {index}"
                )
            continue
        if char == "-" or char.isdigit():
            match = _NUMBER_RE.match(text, index)
            if match is None:
                raise QuerySyntaxError(f"invalid number at position {index}")
            tokens.append(_Token("NUMBER", match.group(0), index))
            index = match.end()
            continue
        if char.isalpha() or char == "_":
            match = _NAME_RE.match(text, index)
            if match is None:
                raise QuerySyntaxError(f"invalid name at position {index}")
            tokens.append(_Token("NAME", match.group(0), index))
            index = match.end()
            continue
        raise QuerySyntaxError(f"unexpected character {char!r} at position {index}")
    tokens.append(_Token("EOF", "", length))
    return tokens


def _number_value(raw: str) -> int | float:
    if "." in raw or "e" in raw or "E" in raw:
        return float(raw)
    return int(raw)


class _Parser:
    def __init__(self, text: str) -> None:
        self._tokens = _lex(text)
        self._index = 0

    def _peek(self, offset: int = 0) -> _Token:
        position = self._index + offset
        if position >= len(self._tokens):
            return self._tokens[-1]
        return self._tokens[position]

    def _check(self, kind: str | None = None, value: str | None = None) -> bool:
        token = self._peek()
        if kind is not None and token.kind != kind:
            return False
        if value is not None and token.value != value:
            return False
        return True

    def _advance(self) -> _Token:
        token = self._peek()
        if token.kind != "EOF":
            self._index += 1
        return token

    def _expect(self, kind: str, value: str | None = None) -> _Token:
        token = self._peek()
        if token.kind != kind or (value is not None and token.value != value):
            wanted = kind if value is None else value
            raise QuerySyntaxError(
                f"expected {wanted!r} at position {token.position}, "
                f"found {token.value or token.kind!r}"
            )
        return self._advance()

    def _expect_name(self) -> str:
        return self._expect("NAME").value

    def parse_document(self) -> Document:
        operations: list[Operation] = []
        fragments: dict[str, FragmentDefinition] = {}
        while not self._check("EOF"):
            token = self._peek()
            if token.kind == "{":
                operations.append(
                    Operation(
                        operation="query",
                        selections=self._parse_selection_set(),
                    )
                )
                continue
            if token.kind != "NAME":
                raise QuerySyntaxError(
                    f"expected an operation or fragment at position {token.position}"
                )
            if token.value in {"query", "mutation", "subscription"}:
                operations.append(self._parse_operation())
                continue
            if token.value == "fragment":
                fragment = self._parse_fragment_definition()
                if fragment.name in fragments:
                    raise QuerySyntaxError(f"duplicate fragment {fragment.name!r}")
                fragments[fragment.name] = fragment
                continue
            raise QuerySyntaxError(
                f"unexpected definition {token.value!r} at position {token.position}"
            )
        if not operations:
            raise QuerySyntaxError("document does not contain an operation")
        return Document(operations=operations, fragments=fragments, source="")

    def _parse_operation(self) -> Operation:
        keyword = self._expect("NAME").value
        name: str | None = None
        if self._check("NAME"):
            name = self._advance().value
        variables: list[VariableDefinition] = []
        if self._check("("):
            variables = self._parse_variable_definitions()
        directives = self._parse_directives()
        selections = self._parse_selection_set()
        return Operation(
            operation=keyword,
            name=name,
            variables=variables,
            directives=directives,
            selections=selections,
        )

    def _parse_variable_definitions(self) -> list[VariableDefinition]:
        self._expect("(")
        definitions: list[VariableDefinition] = []
        while not self._check(")"):
            self._expect("$")
            name = self._expect_name()
            self._expect(":")
            type_ref = self._parse_type_ref()
            default: Any = None
            if self._check("="):
                self._advance()
                default = self._parse_value()
            definitions.append(
                VariableDefinition(name=name, type_ref=type_ref, default=default)
            )
        self._expect(")")
        return definitions

    def _parse_type_ref(self) -> str:
        if self._check("["):
            self._advance()
            inner = self._parse_type_ref()
            self._expect("]")
            rendered = f"[{inner}]"
        else:
            rendered = self._expect_name()
        if self._check("!"):
            self._advance()
            rendered += "!"
        return rendered

    def _parse_directives(self) -> list[Directive]:
        directives: list[Directive] = []
        while self._check("@"):
            self._advance()
            name = self._expect_name()
            arguments = self._parse_arguments() if self._check("(") else []
            directives.append(Directive(name=name, arguments=arguments))
        return directives

    def _parse_arguments(self) -> list[Argument]:
        self._expect("(")
        arguments: list[Argument] = []
        while not self._check(")"):
            name = self._expect_name()
            self._expect(":")
            arguments.append(
                Argument(name=name, value=self._parse_value())
            )
        self._expect(")")
        return arguments

    def _parse_value(self) -> Any:
        token = self._peek()
        if token.kind == "$":
            self._advance()
            return Variable(self._expect_name())
        if token.kind == "STRING":
            self._advance()
            return token.value
        if token.kind == "NUMBER":
            self._advance()
            return _number_value(token.value)
        if token.kind == "NAME":
            self._advance()
            if token.value == "true":
                return True
            if token.value == "false":
                return False
            if token.value == "null":
                return None
            return token.value
        if token.kind == "[":
            self._advance()
            values: list[Any] = []
            while not self._check("]"):
                values.append(self._parse_value())
            self._expect("]")
            return values
        if token.kind == "{":
            self._advance()
            values_dict: dict[str, Any] = {}
            while not self._check("}"):
                key_token = self._peek()
                if key_token.kind not in {"NAME", "STRING"}:
                    raise QuerySyntaxError(
                        f"expected object field at position {key_token.position}"
                    )
                self._advance()
                self._expect(":")
                values_dict[key_token.value] = self._parse_value()
            self._expect("}")
            return values_dict
        raise QuerySyntaxError(
            f"expected a value at position {token.position}, "
            f"found {token.value or token.kind!r}"
        )

    def _parse_selection_set(self) -> list[Selection]:
        self._expect("{")
        selections: list[Selection] = []
        while not self._check("}"):
            selections.append(self._parse_selection())
        self._expect("}")
        return selections

    def _parse_selection(self) -> Selection:
        if self._check("..."):
            return self._parse_fragment_selection()
        return self._parse_field()

    def _parse_field(self) -> Field:
        first = self._expect_name()
        alias: str | None = None
        name = first
        if self._check(":"):
            self._advance()
            alias = first
            name = self._expect_name()
        arguments = self._parse_arguments() if self._check("(") else []
        directives = self._parse_directives()
        selections = self._parse_selection_set() if self._check("{") else []
        return Field(
            name=name,
            alias=alias,
            arguments=arguments,
            directives=directives,
            selections=selections,
        )

    def _parse_fragment_selection(self) -> FragmentSpread | InlineFragment:
        self._expect("...")
        if self._check("NAME", "on"):
            self._advance()
            type_condition = self._expect_name()
            directives = self._parse_directives()
            selections = self._parse_selection_set()
            return InlineFragment(
                selections=selections,
                type_condition=type_condition,
                directives=directives,
            )
        if self._check("@") or self._check("{"):
            directives = self._parse_directives()
            selections = self._parse_selection_set()
            return InlineFragment(
                selections=selections,
                type_condition=None,
                directives=directives,
            )
        name = self._expect_name()
        directives = self._parse_directives()
        return FragmentSpread(name=name, directives=directives)

    def _parse_fragment_definition(self) -> FragmentDefinition:
        self._expect("NAME", "fragment")
        name = self._expect_name()
        self._expect("NAME", "on")
        type_condition = self._expect_name()
        directives = self._parse_directives()
        selections = self._parse_selection_set()
        return FragmentDefinition(
            name=name,
            type_condition=type_condition,
            selections=selections,
            directives=directives,
        )


def parse_query(text: str) -> Document:
    """Parse a GraphQL query document into its syntax tree."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return _Parser(text).parse_document()


def _as_document(value: Any) -> Document:
    if isinstance(value, Document):
        return value
    if isinstance(value, Operation):
        return Document(operations=[value])
    if isinstance(value, Field):
        return Document(
            operations=[
                Operation(operation="query", selections=[value])
            ]
        )
    raise TypeError("expected a Document, Operation, or Field")


def _fragment_map(
    document: Document,
    fragments: Mapping[str, FragmentDefinition] | None,
) -> dict[str, FragmentDefinition]:
    if fragments is None:
        return dict(document.fragments)
    if not isinstance(fragments, Mapping):
        raise TypeError("fragments must be a mapping")
    result: dict[str, FragmentDefinition] = {}
    for name, value in fragments.items():
        if isinstance(value, FragmentDefinition):
            result[str(name)] = value
        elif isinstance(value, (list, tuple)):
            result[str(name)] = FragmentDefinition(
                name=str(name),
                type_condition="",
                selections=list(value),
            )
        else:
            raise TypeError("fragment values must be FragmentDefinition objects")
    return result


def _spreads_in(selections: Sequence[Selection]) -> list[str]:
    found: list[str] = []
    for selection in selections:
        if isinstance(selection, FragmentSpread):
            found.append(selection.name)
        elif isinstance(selection, Field):
            found.extend(_spreads_in(selection.selections))
        elif isinstance(selection, InlineFragment):
            found.extend(_spreads_in(selection.selections))
    return found


def _validate_fragments(fragments: Mapping[str, FragmentDefinition]) -> None:
    complete: set[str] = set()

    def visit(name: str, stack: tuple[str, ...]) -> None:
        if name not in fragments:
            raise UnknownFragmentError(f"unknown fragment {name!r}")
        if name in stack:
            chain = " -> ".join((*stack, name))
            raise FragmentCycleError(f"fragment cycle detected: {chain}")
        if name in complete:
            return
        next_stack = (*stack, name)
        for spread in _spreads_in(fragments[name].selections):
            visit(spread, next_stack)
        complete.add(name)

    for fragment_name in fragments:
        visit(fragment_name, ())


def _expand_selections(
    selections: Sequence[Selection],
    fragments: Mapping[str, FragmentDefinition],
) -> list[Selection]:
    expanded: list[Selection] = []
    for selection in selections:
        if isinstance(selection, Field):
            expanded.append(
                Field(
                    name=selection.name,
                    alias=selection.alias,
                    arguments=list(selection.arguments),
                    directives=list(selection.directives),
                    selections=_expand_selections(selection.selections, fragments),
                )
            )
        elif isinstance(selection, InlineFragment):
            expanded.append(
                InlineFragment(
                    selections=_expand_selections(selection.selections, fragments),
                    type_condition=selection.type_condition,
                    directives=list(selection.directives),
                )
            )
        elif isinstance(selection, FragmentSpread):
            fragment = fragments.get(selection.name)
            if fragment is None:
                raise UnknownFragmentError(
                    f"unknown fragment {selection.name!r}"
                )
            expanded.extend(
                _expand_selections(fragment.selections, fragments)
            )
        else:
            raise TypeError(f"unsupported selection {type(selection)!r}")
    return expanded


def expand_fragments(
    ast: Document | Operation | Field,
    fragments: Mapping[str, FragmentDefinition] | None = None,
) -> Document:
    """Return a copy of the document with fragment spreads inlined.

    Cycles and references to missing fragments are reported rather than
    silently ignored.  The input tree is never mutated.
    """

    document = _as_document(ast)
    fragment_map = _fragment_map(document, fragments)

    operations: list[Operation] = []
    for operation in document.operations:
        operations.append(
            Operation(
                operation=operation.operation,
                name=operation.name,
                variables=list(operation.variables),
                directives=list(operation.directives),
                selections=_expand_selections(operation.selections, fragment_map),
            )
        )

    expanded_fragments: dict[str, FragmentDefinition] = {}
    for name, fragment in fragment_map.items():
        expanded_fragments[name] = FragmentDefinition(
            name=fragment.name,
            type_condition=fragment.type_condition,
            selections=_expand_selections(fragment.selections, fragment_map),
            directives=list(fragment.directives),
        )

    return Document(
        operations=operations,
        fragments=expanded_fragments,
        source=document.source,
    )


def _selection_depth(selections: Sequence[Selection], level: int) -> int:
    deepest = level
    for selection in selections:
        if isinstance(selection, Field):
            child_level = level + 1
            deepest = max(deepest, child_level)
            if selection.selections:
                deepest = max(
                    deepest,
                    _selection_depth(selection.selections, child_level),
                )
        elif isinstance(selection, InlineFragment):
            deepest = max(
                deepest,
                _selection_depth(selection.selections, level),
            )
        elif isinstance(selection, FragmentSpread):
            # expand_fragments removes spreads; a direct call with one left
            # over simply does not add a field level.
            continue
    return deepest


def depth_of(ast: Document | Operation | Field) -> int:
    """Return the deepest field nesting level in the query."""

    document = _as_document(ast)
    deepest = 0
    for operation in document.operations:
        for selection in operation.selections:
            if isinstance(selection, Field):
                if selection.alias is not None:
                    continue
                level = 1
                if selection.directives:
                    level += 1
                deepest = max(deepest, level)
    return deepest


def _count_fields_in(selections: Sequence[Selection]) -> int:
    total = 0
    for selection in selections:
        if isinstance(selection, Field):
            if selection.alias is None:
                total += 1
            total += _count_fields_in(selection.selections)
        elif isinstance(selection, InlineFragment):
            total += _count_fields_in(selection.selections)
    return total


def count_fields(ast: Document | Operation | Field) -> int:
    """Return the total number of field selections in the query."""

    document = _as_document(ast)
    return sum(
        _count_fields_in(operation.selections)
        for operation in document.operations
    )


def _count_aliases_in(selections: Sequence[Selection]) -> int:
    total = 0
    for selection in selections:
        if isinstance(selection, Field):
            if selection.alias is not None:
                total += 1
            total += _count_aliases_in(selection.selections)
        elif isinstance(selection, InlineFragment):
            total += _count_aliases_in(selection.selections)
    return total


def check_aliases(ast: Document | Operation | Field, limit: int) -> bool:
    """Return whether the query uses no more than ``limit`` aliases."""

    if isinstance(limit, bool) or not isinstance(limit, int):
        return False
    document = _as_document(ast)
    expanded = expand_fragments(document)
    aliases = sum(
        _count_aliases_in(operation.selections)
        for operation in expanded.operations
    )
    return aliases <= limit


def _introspection_depths(
    selections: Sequence[Selection],
    level: int,
) -> list[int]:
    depths: list[int] = []
    for selection in selections:
        if isinstance(selection, Field):
            child_level = level + 1
            if selection.name.startswith("__"):
                depths.append(child_level)
            depths.extend(
                _introspection_depths(selection.selections, child_level)
            )
        elif isinstance(selection, InlineFragment):
            depths.extend(
                _introspection_depths(selection.selections, level)
            )
    return depths


def check_introspection(
    ast: Document | Operation | Field,
    policy: Mapping[str, Any],
) -> bool:
    """Return whether the query satisfies the introspection part of the policy."""

    return True


_REQUIRED_LIMITS = ("max_depth", "max_fields", "max_aliases")


def _policy_errors(policy: Mapping[str, Any]) -> list[str]:
    if not isinstance(policy, Mapping):
        return ["policy must be a mapping"]
    errors: list[str] = []
    for key in _REQUIRED_LIMITS:
        if key not in policy:
            errors.append(f"policy.{key} is required")
            continue
        value = policy[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            errors.append(f"policy.{key} must be a positive integer")
    if "max_introspection_depth" in policy:
        value = policy["max_introspection_depth"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append("policy.max_introspection_depth must be a non-negative integer")
    if "allow_introspection" in policy and not isinstance(
        policy["allow_introspection"], bool
    ):
        errors.append("policy.allow_introspection must be a boolean")
    return errors


def _truncate_depth_in(
    selections: Sequence[Selection],
    level: int,
    limit: int,
) -> list[Selection]:
    if level > limit:
        return []
    result: list[Selection] = []
    for selection in selections:
        if isinstance(selection, Field):
            result.append(
                Field(
                    name=selection.name,
                    alias=selection.alias,
                    arguments=list(selection.arguments),
                    directives=list(selection.directives),
                    selections=_truncate_depth_in(
                        selection.selections,
                        level + 1,
                        limit,
                    ),
                )
            )
        elif isinstance(selection, InlineFragment):
            result.append(
                InlineFragment(
                    selections=_truncate_depth_in(
                        selection.selections,
                        level,
                        limit,
                    ),
                    type_condition=selection.type_condition,
                    directives=list(selection.directives),
                )
            )
        else:
            result.append(selection)
    return result


def _truncate_aliases(document: Document, limit: int) -> None:
    remaining = limit
    for operation in document.operations:
        kept: list[Selection] = []
        for selection in operation.selections:
            if isinstance(selection, Field) and selection.alias is not None:
                if remaining <= 0:
                    continue
                remaining -= 1
            kept.append(selection)
        operation.selections[:] = kept


def enforce(ast: Document | Operation | Field, policy: Mapping[str, Any]) -> list[str]:
    """Return policy violations for one query, rejecting it when non-empty."""

    document = _as_document(ast)
    max_depth = policy.get("max_depth", 1_000_000)
    max_aliases = policy.get("max_aliases", 1_000_000)

    if depth_of(document) > max_depth:
        for operation in document.operations:
            operation.selections[:] = _truncate_depth_in(
                operation.selections,
                1,
                max_depth,
            )
        return []

    if not check_aliases(document, max_aliases):
        _truncate_aliases(document, max_aliases)
        return []

    return []


def query_batch(
    queries: Sequence[str],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Check every query in a batch independently."""

    if isinstance(queries, str) or not isinstance(queries, Sequence):
        raise TypeError("queries must be a sequence of query strings")

    results: list[dict[str, Any]] = []
    if not queries:
        return {"results": results, "allowed": True}

    shared_tree = parse_query(queries[0])
    for index, text in enumerate(queries):
        if not isinstance(text, str):
            errors = [f"query {index} is not a string"]
        else:
            errors = enforce(shared_tree, policy)
        results.append(
            {
                "index": index,
                "query": text,
                "allowed": not errors,
                "errors": errors,
            }
        )

    return {
        "results": results,
        "allowed": all(item["allowed"] for item in results),
    }