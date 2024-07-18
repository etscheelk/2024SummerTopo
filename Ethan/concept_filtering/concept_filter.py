import curses
import os
from time import sleep
import pandas as pd
import io

import random

from pyparsing import Path

def get_concepts_from_csv(
    file_path: str | Path
) -> list[str]:
    concepts: pd.DataFrame = pd.read_csv(file_path)
    return list(concepts["concept"])
    pass


def save_blacklist(
    file_path: str | os.PathLike,
    concepts: list[str]
) -> bool:
    path: Path = Path(file_path)
    
    # returns None on success, str on failure
    good: bool = (pd.DataFrame(concepts).to_csv(path) is None)
    
    return good

def create_log() -> io.FileIO:
    fd = open("./concept-filter.log", "+bw")
    return fd

# Micro class because I hate dealing with tuple[0], tuple[1], etc.
# Would prefer if it could just be .min or .max. It's better that way.
class MinMax:
    min: int
    max: int
    def __init__(self, min: int, max: int) -> None:
        self.min = min
        self.max = max
    
    pass

# Custom class to define the range of a window,
# keeping track of bounds, etc.
class WindowRange:
    y_range: range
    x_range: range
    lines: int
    cols: int
    begin_y: int
    begin_x: int
    
    def __init__(self, lines: int, cols: int, begin_y: int, begin_x: int) -> None:
        self.y_range = range(begin_y, begin_y + lines)
        self.x_range = range(begin_x, begin_x + cols)
        self.lines = lines
        self.cols = cols
        self.begin_y = begin_y
        self.begin_x = begin_x
        
        pass
    
    def create_window(self) -> curses.window:
        return curses.newwin(self.lines, self.cols, self.begin_y, self.begin_x)
    
    def min_max_y(self) -> MinMax:
        return MinMax(self.begin_y, self.begin_y + self.lines - 1)
        pass
    
    def min_max_x(self) -> MinMax:
        return MinMax(self.begin_x, self.begin_x + self.cols - 1)
        pass
    
    pass

def main(
    screen: curses.window, 
    blacklist_path: Path,
    provided_concepts: list[str],
    log_file: io.FileIO | None = None 
) -> None:
    # RELEVANCE_MEAN_CUTOFF = 0.7
    # MIN_CONCEPT_OCC = 1
    # MAX_CONCEPT_OCC = 1.
    
    # Write to the log file if it exists with string s
    def add_log(s : str) -> None:
        if log_file is not None:
            log_file.write(bytes(s, "utf8"))
            log_file.flush()
        pass

    concepts = provided_concepts
    
    concepts_blacklist: list[str] = []

    LINES  = curses.LINES  # noqa: F841
    COLS = curses.COLS
    
    curses.curs_set(0)
    
    instruction_range = WindowRange(5, COLS, 0, 0)
    instructions = instruction_range.create_window()
    screen.refresh()
    
    # Draw instructions
    init_line = 1
    style = curses.A_BOLD
    ls = [
        "q to quit",
        "s to save concept list",
        "1-4 to select option"
    ]
    for i in range(0, 3):
        instructions.addstr(init_line+i, 1, ls[i], style)
    instructions.box()
    instructions.refresh()
    
    
    display_range = WindowRange(13, COLS, 10, 0)
    display: curses.window = display_range.create_window()
    screen.refresh()
    
    def draw_default_display(extra_message: str | None = None, displayed_concepts: list[str] | None = None) -> None:
        display.erase()
        display.box()
        
        # Give an extra message at the top-ish line, 
        # like the quit double check
        if extra_message is not None:
            display.addstr(1, 1, extra_message, curses.A_BOLD | curses.A_ITALIC)
            pass
        
        # Draw the displayed concepts, if any
        if displayed_concepts is not None:
            line = 3 # one gap between 
            col = 2
            for i in range(len(displayed_concepts)):
                display.addstr(line+i, col, f"{i+1}: {displayed_concepts[i]}")
                pass
            display.addstr(line+4, col, "4: Pass, show new set", curses.A_ITALIC)
        
        # add counter at the bottom to label the number of concepts and the size of the blacklist
        line = display_range.lines - 3
        col = 1
        display.addstr(line, col, f"Number of concepts: {len(concepts)}")
        display.addstr(line+1, col, f"Number of blacklisted concepts: {len(concepts_blacklist)}")
        
        display.refresh()
        pass
    draw_default_display()

    display_new_concepts: bool = True
    displayed_concepts: list[str] = []
    
    def end_iter() -> None:
        add_log("\n")
        pass
    
    while (True):
        # Run this first in the loop
        # display new concepts, if necessary
        if display_new_concepts and len(concepts) >= 3:
            
            # pick 3 random indices from 
            indices = random.sample(range(len(concepts)), k = 3)
            
            add_log(f"the random sample of indices {indices}\n")
            
            displayed_concepts = [concepts[i] for i in indices]
            
            add_log(f"displayed concepts: {displayed_concepts}\n")
            
            # concepts loaded get shown below
            display_new_concepts = False
            pass
        
        # Run second in the loop
        draw_default_display(
            displayed_concepts = displayed_concepts,
            extra_message = None
        )
        instructions.refresh()
        add_log(f"concepts: {concepts}\n\tsize: {len(concepts)}\n")
        
        key = screen.getkey()
        
        add_log(f"key pressed: {key}\n")
        
        
        if key == "q":
            display.addstr(1, 1, "Did you save the list? Hit q again to quit or any other key to retreat", curses.A_BOLD | curses.A_ITALIC)
            display.refresh()
            if screen.getkey() == "q":
                break
            else:
                end_iter() 
                continue
        if key == "s":
            good: bool = save_blacklist(blacklist_path, concepts_blacklist)
            message: str
            # blacklist_path.relative_to(os.environ["PWD"])
            if good:
                message = f"concepts saved to {blacklist_path}. Press any key to continue."
            else:
                message = f"failure to save concepts to {blacklist_path}. Press any key to continue."
            draw_default_display(extra_message=message, displayed_concepts=displayed_concepts)
            screen.getkey()
            
            end_iter()
            continue
        
        key_int: int = 0
        try:
            key_int = int(key)
            pass
        except:  # noqa: E722
            end_iter(); continue  # noqa: E702
        
        if (key_int in range(1, 3+1)) and len(concepts) > 3:
            rm_ind: int = indices[key_int-1]
            
            add_log(f"the user hit {key_int} to try and remove {concepts[rm_ind]}\n")
            
            concepts_blacklist.append(concepts[rm_ind])
            
            # TODO consider removing ALL displayed concepts
            # Make sure to sort the indices and delete the biggest first
            del concepts[rm_ind]
            display_new_concepts = True
            pass
        if key_int == 4:
            display_new_concepts = True
            pass
        
        end_iter()
        pass
    
    # cleanup()
    
    pass


if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser("Concept Filtering", description="Given a list of concepts, create a blacklist of 'bad' concepts.")
    
    # parser.add_argument("concepts_path", type=str, default=["None"])
    parser.add_argument("--concepts-csv", dest="concepts_csv", type=str, default = None, help="A file path to a csv of concepts. Expects that there is a column titled 'concept'. Holds greater precedence than --concepts.")
    parser.add_argument("--save-path", dest="path", type=str, default = ["./blacklist.csv"], help="Absolute or relative path to save the blacklisted concepts to.", nargs=1)
    parser.add_argument("--concepts", dest="concepts", type=str, nargs="+", help="Provide a list of space separated concepts or else provided by --concepts-csv.")
    parser.add_argument("--test", dest="test", type=bool, default=False, choices=[True, False], help="Test on a sample array. Assumed true if neither --concepts nor --concepts-csv are defined. Holds greatest precedence.", nargs=1)
    parser.add_argument("--log", dest="log", help="Boolean value to save logging information", type=bool, default=False, choices=[True, False], nargs=1)
    
    args = parser.parse_args()
    
    print(f"""Executing from current directory: {os.environ["PWD"]}""")
    
    _log: bool = args.log
    _path = str(args.path[0])
    _test = args.test
    _concepts = args.concepts
    _concepts_csv = args.concepts_csv
    
    log_file: io.FileIO | None = None if not _log else create_log()
    
    
    # check that the concepts csv exists and can be read from,
    # avoiding a big backtrace printed to console
    if _concepts_csv is not None:
        if not Path(_concepts_csv).exists():
            print(f"No such file or directory for the concepts csv at {_concepts_csv}")
            exit()
    
    print("Loading concepts...\n")
    test_concepts: list[str] = ["ethan", "gavin", "frances", "lucia", "lori", "russ", "greg", "kristin", "taylor"]
    
    # check if space sep'd concepts provided
    provided_concepts: list[str] | None = None if _concepts is None else _concepts
    # check if csv provided
    provided_concepts: list[str] | None = \
        provided_concepts if _concepts_csv is None else get_concepts_from_csv(Path(_concepts_csv).absolute())
    provided_concepts: list[str] = test_concepts if provided_concepts is None or _test else provided_concepts
      
    print("Concepts loaded")
    print("{}...".format(provided_concepts if len(provided_concepts) <= 10 else provided_concepts[0:10]))
    
    
    sleep(2)
    
    curses.wrapper(
        main, 
        Path(_path).absolute(),
        provided_concepts,
        log_file
    )
    
    if log_file is not None: 
        log_file.close()