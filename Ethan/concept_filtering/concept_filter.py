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
    
    def min_max_y(self) -> MinMax:
        return MinMax(self.begin_x, self.begin_x + self.cols - 1)
        pass
    
    
    pass


# def write_to_log(s : str, fd: os.BufferedRandom) -> None:
#     os.write(fd, s)

def main(screen: curses.window, log: bool, blacklist_path: str, test: bool) -> None:
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

    # screen.addstr(
    #     f"q to quit\ns to save concept list\n1-3 to select option\n"
    # )
    # screen.refresh()
    
    curses.curs_set(0)
    
    instructions: curses.window = curses.newwin(5, COLS, 0, 0)
    screen.refresh()
    
    instructions.addstr(
        1, 1, f"q to quit\ns to save concept list\n1-3 to select option", curses.A_BLINK
    )
    
    init_line = 1
    style = curses.A_BOLD
    ls = [
        "q to quit",
        "s to save concept list",
        "1-3 to select option"
    ]
    for i in range(0, 3):
        instructions.addstr(init_line+i, 1, ls[i], style)
    # instructions.refresh()
    instructions.box()
    instructions.refresh()
    
    
    # instructions.bkgdset(2)
    
    display_range = WindowRange(10, COLS, 10, 0)
    display: curses.window = display_range.create_window()
    screen.refresh()
    display.box()
    display.refresh()
    

    def draw_default_display() -> None:
        display.erase()
        display.box()
        
        
        
        pass

    display_new_concepts: bool = True
    displayed_concepts: list[str] = []
    def draw_concepts() -> None:
        init_line = 2
        for i in range(len(displayed_concepts)):
            s: str = displayed_concepts[i]
            display.addstr(init_line + i, 2, f"{i+1}: {s}")
            pass
        display.refresh()
        pass
    
    
    
    while (True):
        add_log(f"concepts: {concepts}\n\tsize: {len(concepts)}\n")
        
        # display new concepts, if necessary
        if display_new_concepts and len(concepts) > 3:
            display.erase()
            display.box()
            display.refresh()
            
            # pick 3 random indices from 
            indices = random.sample(range(len(concepts)), k = 3)
            
            add_log(f"the random sample of indices {indices}\n")
            
            displayed_concepts = [concepts[i] for i in indices]
            
            add_log(f"displayed concepts: {displayed_concepts}\n")
            
            draw_concepts()
            display_new_concepts = False
            pass
        
        
        
        key = screen.getkey()
        
        add_log(f"key pressed: {key}\n")
        
        
        if key == "q":
            display.addstr(1, 1, "Did you save the list? Hit q again to quit or any other key to retreat", curses.A_BOLD | curses.A_ITALIC)
            display.refresh()
            if screen.getkey() == "q":
                cleanup()
                break
            else:
                display.erase()
                display.box()
                display.refresh()
                draw_concepts()
                continue
        if key == "s":
            good: bool = save_data(blacklist_path, concepts)
            if good:
                display.addstr(1, 1, f"concepts saved to {blacklist_path}. Press any key to continue", curses.A_BOLD | curses.A_ITALIC)
            else:
                display.addstr(1, 1, f"failure to save concepts to {blacklist_path}. Press any key to continue", curses.A_BOLD | curses.A_ITALIC)
            display.refresh()
            screen.getkey()
            display.erase()
            display.box()
            draw_concepts()
            display.refresh()
            continue
        
        key_int: int = 0
        try:
            key_int = int(key)
            pass
        except:
            continue
        
        if (key_int in range(1, 3+1)) and len(concepts) > 3:
            rm_ind: int = indices[key_int-1]
            
            add_log(f"the user hit {key_int} to try and remove {concepts[rm_ind]}\n")
            
            concepts_blacklist.append(concepts[rm_ind])
            
            del concepts[rm_ind]
            # concepts = concepts[concepts.index != rm_ind].reset_index(drop=True)
            # concepts.reset_index(drop=True)
            display_new_concepts = True
            pass
        
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
    
    curses.wrapper(main, _log, _path, _test)

# end
# curses.nocbreak()
# screen.keypad(False)
# curses.echo()
# curses.endwin()