from typing import List


def line_index(lines: List[str], to_find: str) -> int:
    for (idx, line) in enumerate(lines):
        if to_find in line:
            return idx

    return -1


def get_op_name(line: str) -> str:
    first_sym = line.find('"')
    second_sym = line.find('"', first_sym + 1)
    return line[first_sym + 1:second_sym]


def parse_kinference_file(path: str) -> List[str]:
    with open(path, "r") as ki_file:
        text = ki_file.readlines()
        first_line = line_index(text, "override fun create(name: String, opType: String?, version: Int?, attributes: "
                                      "Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when ("
                                      "opType) {") + 1
        last_line = line_index(text, "else ->")
        ops_lines = text[first_line:last_line]

        return list(map(get_op_name, ops_lines))
