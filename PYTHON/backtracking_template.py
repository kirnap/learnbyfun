"""
Backtracking template used to solve backtracking problems

You may find the original gist from the following link:
https://gist.github.com/RuolinZheng08/cdd880ee748e27ed28e0be3916f56fa6

The video explanation of backtracking is in the following:
https://www.youtube.com/watch?v=A80YzvNwqXA


"""

def is_valid_state(state) -> bool:
	# check if a given state is valid for the solution
	return True


def get_candidates(state) -> List[list]:
	# compute candidates based on rules of a problem / game


def search(state, solutions):
	if is_valid_state(state):
		solutions.append(state.copy()) # Do not forget to take the copy of state!
		return

	for candidate in get_candidates(state):
		state.add(candidate)
		search(state, solutions)
		state.remove(candidate)


def solve():
	solution = []
	state = ()
	search(state, solutions)



