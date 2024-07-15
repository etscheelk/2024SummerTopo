# import curses.panel
import os
import sys
from time import sleep
from more_itertools import collapse
import pandas as pd
import numpy as np
import io

import random

from pyparsing import Path

location = "/home/lalaith/2024SummerTopo/"
sys.path.append(location)
os.chdir(location)

import Gavin.utils.make_network as gutils

import curses


# a: curses._CursesWindow = curses.initscr()

def get_concepts(
    rel_mean_cutoff = 0.7,
    min_articles = 2,
    max_articles = 1.0
) -> list[str]:
    data = gutils.filter_article_concept_file(
        "./datasets/concept_network/concepts_Zoology_608.csv.gz",
        min_relevance=rel_mean_cutoff,
        min_articles=min_articles,
        max_articles=max_articles,
    )
    
    data = data.groupby("concept").size().reset_index()

    data = data["concept"]
    
    return list(data)

def save_data(
    file_path: str | os.PathLike,
    concepts: list[str]
) -> bool:
    path: Path = Path(file_path)
    
    
    # returns None on success, str on failure
    good: bool = (pd.DataFrame(concepts).to_csv(path) == None)
    
    return good

def create_log() -> io.BufferedRandom:
    fd = open("./curses-log.log", "+bw")
    return fd


class MinMax:
    min: int
    max: int
    def __init__(self, min: int, max: int) -> None:
        self.min = min
        self.max = max
    
    pass

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


# def write_to_log(s : str, fd: os.BufferedRandom) -> None:
#     os.write(fd, s)

def main(screen: curses.window, log: bool, blacklist_path: Path, test: bool) -> None:
    RELEVANCE_MEAN_CUTOFF = 0.7
    MIN_CONCEPT_OCC = 1
    MAX_CONCEPT_OCC = 1.

    if log:
        log_file = create_log()
    
    def add_log(s : str) -> None:
        if log:
            log_file.write(bytes(s, "utf8"))
            log_file.flush()
        pass

    def cleanup() -> None:
        if log:
            log_file.close()
        pass

    # screen.clear()

    # concepts = get_concepts()
    # concepts: pd.Series = pd.Series(["ethan", "gavin", "frances", "lucia", "lori", "russ", "greg", "kristin", "taylor"])
    if test:
        concepts: list[str] = ["ethan", "gavin", "frances", "lucia", "lori", "russ", "greg", "kristin", "taylor"]
    else:
        concepts = get_concepts()
    
    concepts_blacklist: list[str] = []
    
    # data = gutils.filter_article_concept_file(
    #     "./datasets/concept_network/concepts_Zoology_608.csv.gz",
    #     min_relevance=RELEVANCE_MEAN_CUTOFF,
    #     min_articles=2,
    #     max_articles=MAX_CONCEPT_OCC,
    # )

    # data = data.groupby("concept").size().reset_index()

    # data = data["concept"]

    # screen = curses.initscr()

    # curses.cbreak(True)

    # screen.keypad(True)

    LINES  = curses.LINES
    COLS = curses.COLS
    
    curses.curs_set(0)
    
    instruction_range = WindowRange(5, COLS, 0, 0)
    instructions = instruction_range.create_window()
    screen.refresh()
    
    # instructions.addstr(
    #     1, 1, f"q to quit\ns to save concept list\n1-3 to select option", curses.A_BLINK
    # )
    
    init_line = 1
    style = curses.A_BOLD
    ls = [
        "q to quit",
        "s to save concept list",
        "1-4 to select option"
    ]
    for i in range(0, 3):
        instructions.addstr(init_line+i, 1, ls[i], style)
    # instructions.refresh()
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
            # if len(displayed_concepts) > 3:
            display.addstr(line+4, col, f"4: Pass, show new set", curses.A_ITALIC)
            # else:
                # display.addstr(line+4, col, f"No more concepts to show", curses.A_BOLD | curses.A_ITALIC)
        
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
            
            # draw_concepts()
            # draw_default_display(displayed_concepts=displayed_concepts)
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
                end_iter(); continue
        if key == "s":
            good: bool = save_data(blacklist_path, concepts_blacklist)
            message: str
            # blacklist_path.relative_to(os.environ["PWD"])
            if good:
                message = f"concepts saved to {blacklist_path}. Press any key to continue."
            else:
                message = f"failure to save concepts to {blacklist_path}. Press any key to continue."
            draw_default_display(extra_message=message, displayed_concepts=displayed_concepts)
            screen.getkey()
            
            end_iter(); continue
        
        key_int: int = 0
        try:
            key_int = int(key)
            pass
        except:
            end_iter(); continue
        
        if (key_int in range(1, 3+1)) and len(concepts) > 3:
            rm_ind: int = indices[key_int-1]
            
            add_log(f"the user hit {key_int} to try and remove {concepts[rm_ind]}\n")
            
            concepts_blacklist.append(concepts[rm_ind])
            
            del concepts[rm_ind]
            # concepts = concepts[concepts.index != rm_ind].reset_index(drop=True)
            # concepts.reset_index(drop=True)
            display_new_concepts = True
            pass
        if key_int == 4:
            display_new_concepts = True
            pass
        
        end_iter()
        pass
    
    cleanup()
    
    pass


if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser("Concept Filtering", description="Given a list of concepts, create a blacklist of 'bad' concepts.")
    
    # parser.add_argument("concepts_path", type=str, default=["None"])
    parser.add_argument("--log", dest="log", help="Boolean value to save logging information", type=bool, default=False, choices=[True, False], nargs=1)
    parser.add_argument("--path", dest="path", type=str, default = ["./blacklist.csv"], help="The path to save the blacklisted concepts to", nargs=1)
    parser.add_argument("--test", dest="test", type=bool, default=False, choices=[True, False], help="Test on a sample array", nargs=1)
    
    
    args = parser.parse_args()
    
    # print(os.environ["PWD"])
    
    a: list[str] = ["bob"]
    a = collapse(a)
    print(a)
    
    _log: bool = args.log
    _path = str(args.path[0])
    _test = args.test
    
    print(args)
    
    # print(_path.absolute())
    
    
    # p = Path("./")
    sleep(2)
    
    curses.wrapper(main, _log, Path(_path).absolute(), _test)