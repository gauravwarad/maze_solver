# maze_solver
CPSC-481 project


Files:
    requirements.txt
        libraries installed
    
    theratandthemaze.py
        the main file with all the code

    execution_times_20.csv
        sample output file
    
    mouse.png
        using mouse as pointer, this has to be in the directory

Structure:

Timer Class

    start: Starts the timer.
    stop: Stops the timer and calculates the elapsed time.
    get_elapsed_time: Returns the elapsed time in seconds.
    reset: Resets the timer.

MazeGenerator Class
    Handles maze generation, visualization, and search algorithms.

    Initialization:
        Creates a 2D grid for the maze.
        Tracks visited cells and stores the maze's path.

    Maze Generation:
        generate_maze: Uses recursive backtracking to create a maze.
        get_direction_bit: Encodes direction into integer values.
        add_entrance_and_exit: Adds entrance at the top-left and exit at the bottom-right.
        reset_for_new_search: Resets visited cells and paths for a new search.

    Pathfinding Algorithms:
        a_star: Implements the A* algorithm to find the shortest path.
        depth_first_search: Uses DFS to explore the maze.
        uniform_cost_search: Implements UCS for pathfinding.
        depth_limited_search: Performs DFS with a depth limit.

    Helper Functions:
        is_wall_between: Checks if there is a wall between two cells.
        reconstruct_path: Rebuilds the path from the search algorithm.

Statistics and Visualization
    save_statistics:
        Saves search algorithm statistics (e.g., time, visited cells) to a CSV file.
    show_statistics:
        Displays saved statistics in a readable format.
    visualize_search:
        Animates the pathfinding process using matplotlib.



To install required libraries:
    run the command
    pip install -r requirements.txt

To run the code,
    python theratandthemaze.py

    if your system needs python3 to be specified, try

    python3 theratandthemaze.py

Code has three options, press the required operation key and hit enter.
    for visualization, specify the maze size and hit enter.
        the visualization windows should pop up showing visualizations.
        close the window to view the next visualization.
        keep an eye on the command line to see the algorithm, time & number of visited node statistics.
    
    for the option to show statistics, 
        maze size can be given as input
        by default it's being averaged by running 1000 times which can be easily changed in the for loop
    
    save statistics
        same as show statistics,
        run time statistics are saved to a csv file