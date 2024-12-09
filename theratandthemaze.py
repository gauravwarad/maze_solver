import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg
import heapq
import time
import csv
from pathlib import Path

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None

    def start(self):
        """Starts the timer."""
        self.start_time = time.time()
        self.end_time = None
        self.elapsed_time = None

    def stop(self):
        """Stops the timer and calculates elapsed time."""
        if self.start_time is None:
            raise ValueError("Timer has not been started yet.")
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time

    def get_elapsed_time(self):
        """Returns the elapsed time in seconds."""
        if self.elapsed_time is None:
            raise ValueError("Timer has not been stopped yet.")
        return self.elapsed_time

    def reset(self):
        """Resets the timer."""
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        
#####################################################

class MazeGenerator:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.maze = [[0 for _ in range(width)] for _ in range(height)]
        self.visited = [[False for _ in range(width)] for _ in range(height)]
        self.path = []  # To store the final path
        self.visited_cells = []  # To store all visited cells during the search
        
    def generate_maze(self, x=0, y=0):
        self.visited[y][x] = True
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height and not self.visited[ny][nx]:
                self.maze[y][x] |= self.get_direction_bit(dx, dy)
                self.maze[ny][nx] |= self.get_direction_bit(-dx, -dy)
                self.generate_maze(nx, ny)

    def get_direction_bit(self, dx, dy):
        if dx == 1: return 1
        if dx == -1: return 2
        if dy == 1: return 4
        if dy == -1: return 8

    def add_entrance_and_exit(self):
        self.maze[0][0] |= 8
        self.maze[self.height - 1][self.width - 1] |= 4

    def reset_for_new_search(self):     ##ADDED
        self.visited = [[False for _ in range(self.width)] for _ in range(self.height)]  # Reset visited state
        self.visited_cells = []  # Clear visited cells for visualization
        self.path = []  # Clear the path

    def a_star(self, start, end):
        timer = Timer()
        timer.start()
        def heuristic(a, b):
            return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

        # Initialize the frontier with the initial state
        frontier = []
        heapq.heappush(frontier, (0, start))  # Priority queue with (cost, node)

        # Initialize explored set
        explored = set()

        # Cost dictionaries
        came_from = {}
        g_score = {start: 0}  # Cost to reach the node
        f_score = {start: heuristic(start, end)}  # Estimated total cost (g + h)

        while frontier:
            # If the frontier is empty, return FAILURE (handled implicitly)
            _, current = heapq.heappop(frontier)

            # Mark as visited for visualization
            self.visited_cells.append(current)

            # Check if goal is reached
            if current == end:
                self.reconstruct_path(came_from, current)
                timer.stop()
                print(f"A* Time: {timer.get_elapsed_time():.9f} seconds")
                return True

            # Add current node to the explored set
            explored.add(current)

            # For every child node of the current node
            x, y = current
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if not self.is_wall_between((x, y), (nx, ny)):
                        neighbor = (nx, ny)
                        temp_g_score = g_score[current] + 1

                        # Check if the neighbor is already explored or on the frontier
                        if neighbor not in explored and (
                            neighbor not in g_score or temp_g_score < g_score[neighbor]
                        ):
                            # Update costs and add to frontier
                            came_from[neighbor] = current
                            g_score[neighbor] = temp_g_score
                            f_score[neighbor] = temp_g_score + heuristic(neighbor, end)

                            # Replace if already in frontier with a higher cost
                            in_frontier = any(neighbor == node for _, node in frontier)
                            if not in_frontier:
                                heapq.heappush(frontier, (f_score[neighbor], neighbor))
                            else:
                                # Remove the old entry and push the updated one
                                frontier = [
                                    item for item in frontier if item[1] != neighbor
                                ]
                                heapq.heapify(frontier)
                                heapq.heappush(frontier, (f_score[neighbor], neighbor))

        # If the loop exits without finding a solution, return FAILURE
        timer.stop() # stops timer if stuck, no path found
        print(f"A* Time: {timer.get_elapsed_time():.9f} seconds") 
        return False
    
    def depth_first_search(self, start, end):
        timer = Timer()
        timer.start()
        
        stack = [start]  # stack
        came_from = {}  # reconstruct the path
        self.visited_cells = []  # Reset visited cells for visualization
        self.path = []  # Reset the path for visualization

        visited = set()  # track visited nodes, prevent infinite looping 
        visited.add(start)  # Mark the start as visited
        while stack: 
            current = stack.pop()
            self.visited_cells.append(current)

            if current == end:
                self.reconstruct_path(came_from, current)
                timer.stop()
                print(f"DFS Time: {timer.get_elapsed_time():.9f} seconds")
                return True
            
            x, y = current 
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

            for dx, dy in directions: 
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbor = (nx, ny)
                    if neighbor not in visited and not self.is_wall_between(current, neighbor):
                        stack.append(neighbor)
                        visited.add(neighbor)
                        came_from[neighbor] = current
        timer.stop()
        print(f"DFS Time: {timer.get_elapsed_time():.9f} seconds")
        return False

    def uniform_cost_search(self, start, end):
        timer = Timer()
        timer.start()

        frontier =[]
        heapq.heappush(frontier, (0, start))

        visited = set()

        came_from = {}
        g_score = {start: 0}

        while frontier:
            current_cost, current = heapq.heappop(frontier)
            self.visited_cells.append(current)

            if current == end:
                self.reconstruct_path(came_from, current)
                timer.stop()
                print(f"UCS Time: {timer.get_elapsed_time():.9f} seconds")    
                return True
            visited.add(current)

            x, y = current
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if not self.is_wall_between((x, y), (nx, ny)):
                        neighbor = (nx, ny)
                        new_cost = current_cost + 1

                        # If the neighbor is not visited or has a lower cost
                        if neighbor not in visited or new_cost < g_score.get(neighbor, float('inf')):
                            came_from[neighbor] = current
                            g_score[neighbor] = new_cost
                            heapq.heappush(frontier, (new_cost, neighbor))

        timer.stop()
        print(f"UCS Time: {timer.get_elapsed_time():.9f} seconds")
        return False

    def reconstruct_path(self, came_from, current):
        while current in came_from:
            self.path.append(current)
            current = came_from[current]
        self.path.reverse()  # Reverse to start from the beginning

    def is_wall_between(self, cell1, cell2):
        x1, y1 = cell1
        x2, y2 = cell2
        dx, dy = x2 - x1, y2 - y1
        if dx == 1: return not self.maze[y1][x1] & 1
        if dx == -1: return not self.maze[y1][x1] & 2
        if dy == 1: return not self.maze[y1][x1] & 4
        if dy == -1: return not self.maze[y1][x1] & 8
        return True

    def visualize_pathfinding(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.axis("off")

        for y in range(self.height):
            for x in range(self.width):
                cell = self.maze[y][x]
                if not cell & 8: ax.plot([x, x + 1], [self.height - y, self.height - y], color="black")
                if not cell & 1: ax.plot([x + 1, x + 1], [self.height - y - 1, self.height - y], color="black")
                if not cell & 2: ax.plot([x, x], [self.height - y - 1, self.height - y], color="black")
                if not cell & 4: ax.plot([x, x + 1], [self.height - y - 1, self.height - y - 1], color="black")
        # cheese at the bottom right
        #todo: ---

        visited_cells, = ax.plot([], [], 'o', color="lightgreen", markersize=6)
        final_path, = ax.plot([], [], '-', color="darkgreen", lw=2)
        agent_image = mpimg.imread('mouse.png')
        agent = ax.imshow(agent_image, extent=(0.4, 0.6, 0.4, 0.6), zorder=5)

        total_frames = len(self.visited_cells) + len(self.path)
        # pass    #added

        

        def update(frame):
            if frame < len(self.visited_cells):
                # Update visited cells
                visited_x, visited_y = zip(*self.visited_cells[:frame + 1])
                visited_cells.set_data([x + 0.5 for x in visited_x],
                                    [self.height - y - 0.5 for y in visited_y])
            elif frame - len(self.visited_cells) < len(self.path):
                # Update the final path and agent after all visited nodes are marked
                path_index = frame - len(self.visited_cells)
                path_x, path_y = zip(*self.path[:path_index + 1])
                final_path.set_data([x + 0.5 for x in path_x],
                                    [self.height - y - 0.5 for y in path_y])

                current = self.path[path_index]
                agent.set_extent((current[0], current[0] + 1, self.height - current[1] - 1, self.height - current[1]))
            return visited_cells, final_path, agent

        ani = FuncAnimation(fig, update, frames=total_frames, interval=100, blit=True, repeat=False)
        plt.show()

def save_statistics():
    for _ in range(1000):
        width, height = 30, 30
        maze_gen = MazeGenerator(width, height)
        maze_gen.generate_maze()
        maze_gen.add_entrance_and_exit()

        dfstime = Timer()
        dfstime.start()
        if maze_gen.depth_first_search((0, 0), (width - 1, height - 1)):
            # maze_gen.visualize_pathfinding()
            dfstime.stop()
        else:
            print("No DFS path found!")

        maze_gen.reset_for_new_search()

        astartime = Timer()
        astartime.start()
        if maze_gen.a_star((0, 0), (width - 1, height - 1)):
            # maze_gen.visualize_pathfinding()
            astartime.stop()
        else:
            print("No A-star path found!")


        maze_gen.reset_for_new_search()

        usctime = Timer()
        usctime.start()
        if maze_gen.uniform_cost_search((0, 0), (width - 1, height - 1)):
            # maze_gen.visualize_pathfinding()
            usctime.stop()
        else:
            print("No UCS path found!")
        
        save_to_file(dfstime.get_elapsed_time(), astartime.get_elapsed_time(), usctime.get_elapsed_time() )


def save_to_file(dfs_time, astar_time, usc_time, filename="execution_times_30.csv"):
    # Check if the file exists
    file_exists = Path(filename).is_file()
    # Open the file in append mode and write data
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the header if the file does not exist
        if not file_exists:
            writer.writerow(["DFS Time (s)", "A* Time (s)", "UCS Time (s)"])
        # Write the times to a new row
        writer.writerow([dfs_time, astar_time, usc_time])
    print("file written")

# save_statistics()
# Usage
# uncomment below to run.
width, height = 20, 20
maze_gen = MazeGenerator(width, height)
maze_gen.generate_maze()
maze_gen.add_entrance_and_exit()

if maze_gen.depth_first_search((0, 0), (width - 1, height - 1)):
    maze_gen.visualize_pathfinding()
else:
    print("No DFS path found!")

maze_gen.reset_for_new_search()

if maze_gen.a_star((0, 0), (width - 1, height - 1)):
    maze_gen.visualize_pathfinding()
else:
    print("No A-star path found!")


maze_gen.reset_for_new_search()

if maze_gen.uniform_cost_search((0, 0), (width - 1, height - 1)):
    maze_gen.visualize_pathfinding()
else:
    print("No UCS path found!")


