"""Logic for processing SQL"""

import re
from typing import Sequence

import sqlparse


def _create_pattern(syntax: Sequence[str]) -> str:
    whitespace = r"[\s\n]*"
    return re.compile(whitespace + whitespace.join(syntax) + whitespace)


STUFF = r"([^\s][\s\n]*)*"
SELECT_SYNTAX = _create_pattern(("select", STUFF, "from", STUFF))
INSERT_SYNTAX = _create_pattern(("insert", "into", STUFF, "values", STUFF))
UPDATE_SYNTAX = _create_pattern(("update", STUFF, "set", STUFF))
DELETE_SYNTAX = _create_pattern(("delete", "from", STUFF, "where", STUFF))
INSERT_INTO_SELECT_SYNTAX = _create_pattern(
    ("insert", "into", STUFF, "select", STUFF, "from", STUFF)
)
CREATE_TABLE_SYNTAX = _create_pattern(("create", "table", STUFF))
CREATE_VIEW_SYNTAX = _create_pattern(("create", "view", STUFF))
CREATE_FUNCTION_SYNTAX = _create_pattern(("create", "function", STUFF))
CREATE_OR_REPLACE_TABLE_SYNTAX = _create_pattern(("create", "or", "replace", "table", STUFF))
CREATE_OR_REPLACE_VIEW_SYNTAX = _create_pattern(("create", "or", "replace", "view", STUFF))
CREATE_OR_REPLACE_FUNCTION_SYNTAX = _create_pattern(("create", "or", "replace", "function", STUFF))
ALTER_TABLE_SYNTAX = _create_pattern(("alter", "table", STUFF))
CREATE_INDEX_SYNTAX = _create_pattern(("create", "index", STUFF, "on", STUFF))
CREATE_UNIQUE_INDEX_SYNTAX = _create_pattern(("create", "unique", "index", STUFF, "on", STUFF))
DROP_TABLE_SYNTAX = _create_pattern(("drop", "table", STUFF))
DROP_VIEW_SYNTAX = _create_pattern(("drop", "view", STUFF))


def format_sql(content: str) -> str:
    return sqlparse.format(content, reindent=True, keyword_case='lower')


def is_sql_syntax(content: str) -> bool:
    return any(
        syntax.match(content.lower())
        for syntax in (
            SELECT_SYNTAX,
            INSERT_SYNTAX,
            UPDATE_SYNTAX,
            DELETE_SYNTAX,
            INSERT_INTO_SELECT_SYNTAX,
            CREATE_TABLE_SYNTAX,
            CREATE_VIEW_SYNTAX,
            CREATE_FUNCTION_SYNTAX,
            CREATE_OR_REPLACE_TABLE_SYNTAX,
            CREATE_OR_REPLACE_VIEW_SYNTAX,
            CREATE_OR_REPLACE_FUNCTION_SYNTAX,
            ALTER_TABLE_SYNTAX,
            CREATE_INDEX_SYNTAX,
            CREATE_UNIQUE_INDEX_SYNTAX,
            DROP_TABLE_SYNTAX,
            DROP_VIEW_SYNTAX,
        )
    )
