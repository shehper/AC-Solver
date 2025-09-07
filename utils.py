import ast

def normalize_input(line, relatorLen):
    lineLen = len(line)
    assert lineLen % 2 == 0
    genLen = len(line)//2
    if relatorLen <= genLen:
        return line
    
    padding = relatorLen - genLen
    gen1, gen2 = line[: genLen], line[genLen:]
    gen1 = gen1 + [0]*padding
    gen2 = gen2 + [0]*padding
    return gen1+gen2

def read_all_lines(filename):
    lines = []
    with open(filename, 'r') as file:
        for line in file:
            if line := line.strip():  # skip empty lines
                try:
                    parsed_list = ast.literal_eval(line)
                    if isinstance(parsed_list, list):
                        lines.append([int(x) for x in parsed_list])
                except (ValueError, SyntaxError):
                    continue  # skip invalid lines
    return lines

def read_list_file(filename, relatorLen=0):
    lines = read_all_lines(filename)
    maxLen = relatorLen
    for line in lines:
        line_len = len(line)
        assert (line_len % 2 == 0)
        maxLen = max(maxLen, line_len//2)
    for i, line in enumerate(lines):
        lines[i] = normalize_input(line, maxLen)
    return lines, maxLen
