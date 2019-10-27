class CNF:
    def __init__(self, n_variables, clauses, occur_list, comments):
        self.n_variables = n_variables
        self.clauses = clauses
        self.occur_list = occur_list
        self.comments = comments

    def __str__(self):
        return f"""Number of variables: {self.n_variables}
Clauses: {str(self.clauses)}
Comments: {str(self.comments)}"""

    def to_file(self, filename):
        with open(filename, 'w') as f:
            f.write(self.to_string())

    def to_string(self):
        string = f'p cnf {self.n_variables} {len(self.clauses)}\n'
        for clause in self.clauses:
            string += ' '.join(str(literal) for literal in clause) + ' 0\n'
        return string

    @classmethod
    def from_file(cls, filename):
        with open(filename) as f:
            return cls.from_string(f.read())

    @classmethod
    def from_string(cls, string):
        n_variables, clauses, occur_list, comments = CNF.parse_dimacs(string)
        return cls(n_variables, clauses, occur_list, comments)

    @staticmethod
    def parse_dimacs(string):
        n_variables = 0
        clauses = []
        comments = []
        for line in string.splitlines():
            line = line.strip()
            if not line:
                continue
            elif line[0] == 'c':
                comments.append(line)
            elif line.startswith('p cnf'):
                tokens = line.split()
                n_variables, n_remaining_clauses = int(tokens[2]), int(tokens[3])
                occur_list = [[] for _ in range(n_variables * 2 + 1)]
            elif n_remaining_clauses > 0:
                clause = []
                clause_index = len(clauses)
                for literal in line.split()[:-1]:
                    literal = int(literal)
                    clause.append(literal)
                    occur_list[literal].append(clause_index)
                clauses.append(clause)
                n_remaining_clauses -= 1
            else:
                break
        return n_variables, clauses, occur_list, comments
