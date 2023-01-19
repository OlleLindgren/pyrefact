#!/usr/bin/env python3

import sys

from pyrefact import sql


def main() -> int:
    assert sql.SELECT_SYNTAX.match(
        """
        select *
        from files_from_vacation_with_auntie
        """.lower(),
    )
    assert sql.INSERT_SYNTAX.match(
        """
        insert into users (x, y, z)
        values ('u', 'v', 3)
        """.lower(),
    )
    assert sql.UPDATE_SYNTAX.match("""
                                   update links
                                   set qkf=19
                                   where false
                                   """.lower())
    assert sql.DELETE_SYNTAX.match("""
                                   delete
                                   from temp_table_3
                                   where true;
                                   """.lower())
    assert sql.INSERT_INTO_SELECT_SYNTAX.match(
        """
        insert into dogs (name)
        select name
        from horses
        """.lower(),
    )
    assert sql.CREATE_TABLE_SYNTAX.match(
        """
        create table q (col1 int4 primary ky,
                                          col2 int8 not null)
        """.lower(),
    )
    assert sql.CREATE_VIEW_SYNTAX.match("""
                                        create view f as
                                        select *
                                        from images
                                        """.lower())
    assert sql.CREATE_FUNCTION_SYNTAX.match(
        """
        create function x as
        select y
        from z
        """.lower(),
    )
    assert sql.CREATE_OR_REPLACE_TABLE_SYNTAX.match(
        'create or replace table w (col1 int4 primary ky, col2 int8 not null)'.lower(),
    )
    assert sql.CREATE_OR_REPLACE_VIEW_SYNTAX.match(
        """
        create or replace view f as
        select *
        from images
        """.lower(),
    )
    assert sql.CREATE_OR_REPLACE_FUNCTION_SYNTAX.match(
        """
        create or replace function x as
        select y
        from z
        """.lower(),
    )
    assert sql.ALTER_TABLE_SYNTAX.match(
        "alter table dogs add column weight int8 not null;".lower(),
    )
    assert sql.CREATE_INDEX_SYNTAX.match(
        "create index ix on dogs (name, height)".lower(),
    )
    assert sql.CREATE_UNIQUE_INDEX_SYNTAX.match(
        'create unique index ix on dogs (name, height)'.lower(),
    )
    assert sql.DROP_TABLE_SYNTAX.match(
        "drop table some_nice_table_that_we_like_a_lot;".lower(),
    )
    assert sql.DROP_VIEW_SYNTAX.match(
        "drop view some_nice_view_that_we_like_a_lot;".lower(),
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
