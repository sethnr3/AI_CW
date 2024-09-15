import random
import statistics

class Maze:
    def __init__(self, maze):
        self.maze = maze
        self.rows = len(maze)
        self.cols = len(maze[0])

    def is_valid_move(self, row, col):
        return 0 <= row < self.rows and 0 <= col < self.cols and self.maze[row][col] != 3

    def get_neighbors(self, row, col):
        neighbors = []
        directions = [
            (0, 1), (1, 0), (0, -1), (-1, 0),  # right, down, left, up
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # diagonals
        ]
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self.is_valid_move(new_row, new_col):
                neighbors.append((new_row, new_col))
        return neighbors

class DFSResult:
    def __init__(self):
        self.visited = set()
        self.path = []
        self.time_to_goal = 0
        self.path_length = 0

def dfs_recursive(maze, current, goal, result):
    result.visited.add(current)
    result.path.append(current)

    if current == goal:
        result.time_to_goal = len(result.visited)
        return True

    neighbors = sorted(maze.get_neighbors(current[0], current[1]))
    for neighbor in neighbors:
        if neighbor not in result.visited and dfs_recursive(maze, neighbor, goal, result):
            return True

    result.path.pop()
    return False

def perform_dfs(maze, start, goal):
    result = DFSResult()
    dfs_recursive(maze, start, goal, result)
    result.path_length = len(result.path)
    return result

def manhattan_heuristic(node, goal):
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

class AStarResult:
    def __init__(self):
        self.visited = set()
        self.path = []
        self.time_to_goal = 0
        self.path_length = 0

def astar_search(maze, start, goal):
    result = AStarResult()
    open_set = [(start, 0)]  # Priority queue with initial cost

    while open_set:
        current, cost_so_far = open_set.pop(0)
        result.visited.add(current)
        result.path.append(current)

        if current == goal:
            result.time_to_goal = cost_so_far
            result.path_length = len(result.path)
            return result

        neighbors = maze.get_neighbors(current[0], current[1])
        for neighbor in neighbors:
            if neighbor not in result.visited:
                heuristic_cost = manhattan_heuristic(neighbor, goal)
                total_cost = cost_so_far + 1 + heuristic_cost  # 1 is the cost to move to a neighbor
                open_set.append((neighbor, total_cost))
                open_set.sort(key=lambda x: x[1])  # Sort by total cost

    return result

def generate_random_maze():
    maze_data = [[0] * 6 for _ in range(6)]

    start_row = random.randint(0, 3)
    start_col = random.randint(0, 1)
    start_point = (start_row, start_col)

    goal_row = random.randint(2, 5)
    goal_col = random.randint(4, 5)
    goal_point = (goal_row, goal_col)

    available_nodes = set(range(36)) - {start_point[0] * 6 + start_point[1], goal_point[0] * 6 + goal_point[1]}
    barrier_nodes = random.sample(list(available_nodes), 4)

    maze_data[start_row][start_col] = 1
    maze_data[goal_row][goal_col] = 2
    for barrier_node in barrier_nodes:
        barrier_row, barrier_col = divmod(barrier_node, 6)
        maze_data[barrier_row][barrier_col] = 3

    return maze_data

def analyze_results(results, algorithm_name):
    completeness = all(result.time_to_goal > 0 for result in results)
    optimality = all(result.time_to_goal == result.path_length for result in results)
    solution_times = [result.time_to_goal for result in results]
    path_lengths = [result.path_length for result in results]

    print(f"\nAnalysis for {algorithm_name}:")
    print(f"Completeness: {completeness}")
    print(f"Optimality: {optimality}")
    print(f"Mean Solution Time: {statistics.mean(solution_times)} minutes")
    print(f"Variance in Solution Time: {statistics.variance(solution_times)}")


# Repeat the process for three random mazes
dfs_results = []
astar_results = []

for _ in range(3):
    maze_data = generate_random_maze()
    maze = Maze(maze_data)

    start_point = (random.randint(0, 3), random.randint(0, 1))
    goal_point = (random.randint(2, 5), random.randint(4, 5))

    # Perform DFS
    dfs_result = perform_dfs(maze, start_point, goal_point)
    dfs_results.append(dfs_result)

    # Perform A*
    astar_result = astar_search(maze, start_point, goal_point)
    astar_results.append(astar_result)

    # Analyze and print results for each algorithm
    print("\nMaze Data:")
    for row in maze_data:
        print(row)

    print("\nDFS Results:")
    print("Time to Find Goal:", dfs_result.time_to_goal, "minutes")
    print("Final Path:", dfs_result.path)

    print("\nA* Search Results:")
    print("Time to Find Goal:", astar_result.time_to_goal, "minutes")
    print("Final Path:", astar_result.path)

# Analyze and report overall results
analyze_results(dfs_results, "DFS")
analyze_results(astar_results, "A*")
