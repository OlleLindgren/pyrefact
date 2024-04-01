#!/usr/bin/env python3


import sys
from pathlib import Path

from pyrefact import fixes

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = ((
        """
import requests

session = requests.Session()
session.get("https://www.google.com")
        """,
        """
import requests

with requests.Session() as session:
    session.get("https://www.google.com")
        """,
        ),
        (
            """
x = open("path/to/file.py")
print(x.read())

x.close()
        """,
        """
with open("path/to/file.py") as x:
    print(x.read())
        """,
        ),
        (
            """
with open("path/to/file.py") as x:
    print(x.read())
        """,
            """
with open("path/to/file.py") as x:
    print(x.read())
        """,
        ),
        (
            """
@app.route('/capacities', methods=['GET'])
@cross_origin()
def get_capacity():
    connection = psycopg2.connect(DATABASE_URI)
    cursor = connection.cursor()
    cursor.execute("SELECT id, name FROM capacity")
    capacities = [{"id": row[0], "name": row[1]} for row in cursor.fetchall()]

    return jsonify(capacities)
        """,
            """
@app.route('/capacities', methods=['GET'])
@cross_origin()
def get_capacity():
    with psycopg2.connect(DATABASE_URI) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT id, name FROM capacity")
            capacities = [{"id": row[0], "name": row[1]} for row in cursor.fetchall()]

            return jsonify(capacities)
        """,
    ),)

    for source, expected_abstraction in test_cases:
        processed_content = fixes.missing_context_manager(source)

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
