# import curses.panel
import os
import sys
import pandas as pd
import numpy as np
import io

import random

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
    # good: bool = type(concepts.to_csv(file_path) == None)
    good = True # TODO check return value
    pd.DataFrame(concepts).to_csv(file_path)
    
    return good

def create_log() -> io.BufferedRandom:
    fd = open("./curses-log.log", "+bw")
    return fd





# def write_to_log(s : str, fd: os.BufferedRandom) -> None:
#     os.write(fd, s)

def main(screen: curses.window) -> None:
    RELEVANCE_MEAN_CUTOFF = 0.7
    MIN_CONCEPT_OCC = 1
    MAX_CONCEPT_OCC = 1.

    log_file = create_log()
    

    # screen.clear()
    
    print(type(screen))

    # concepts = get_concepts()
    # concepts: pd.Series = pd.Series(["ethan", "gavin", "frances", "lucia", "lori", "russ", "greg", "kristin", "taylor"])
    concepts: list[str] = ["ethan", "gavin", "frances", "lucia", "lori", "russ", "greg", "kristin", "taylor"]
    concepts = get_concepts()
    # while
    
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
    
    display: curses.window = curses.newwin(10, COLS, 10, 0)
    screen.refresh()
    display.box()
    display.refresh()

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
        log_file.write(bytes(f"concepts: {concepts}\n\tsize: {len(concepts)}\n", "utf8"))
        
        # display new concepts, if necessary
        if display_new_concepts and len(concepts) >= 3:
            display.erase()
            display.box()
            display.refresh()
            # pick 3 random indices from 
            indices = random.sample(range(len(concepts)), k = 3)
            
            log_file.write(bytes(f"the random sample of indices {indices}\n", "utf8"))
            
            displayed_concepts = [concepts[i] for i in indices]
            
            log_file.write(bytes(f"displayed concepts: {displayed_concepts}\n", "utf8"))
            
            draw_concepts()
            display_new_concepts = False
            pass
        
        
        
        key = screen.getkey()
        log_file.write(bytes(f"key pressed: {key}\n", "utf8"))
        log_file.flush()
        
        
        
        
        if key == "q":
            display.addstr(1, 1, "Did you save the list? Hit q again to quit or any other key to retreat", curses.A_BOLD | curses.A_ITALIC)
            display.refresh()
            if screen.getkey() == "q":
                log_file.close()
                break
            else:
                display.erase()
                display.box()
                display.refresh()
                draw_concepts()
                continue
        if key == "s":
            path = "./test.csv"
            good: bool = save_data(path, concepts)
            if good:
                display.addstr(1, 1, f"concepts saved to {path}. Press any key to continue", curses.A_BOLD | curses.A_ITALIC)
            else:
                display.addstr(1, 1, f"failure to save concepts to {path}. Press any key to continue", curses.A_BOLD | curses.A_ITALIC)
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
            pass
        
        if (key_int in range(1, 3+1)) and len(concepts) >= 3:
            rm_ind: int = indices[key_int-1]
            
            log_file.write(bytes(f"the user hit {key_int} to try and remove {concepts[rm_ind]}\n", "utf8"))
            
            del concepts[rm_ind]
            # concepts = concepts[concepts.index != rm_ind].reset_index(drop=True)
            # concepts.reset_index(drop=True)
            display_new_concepts = True
            pass
            
            
        

        
        # pick 3 concepts and print them
        
        # if screen.getkey():
        #     break
        pass

curses.wrapper(main)

# end
# curses.nocbreak()
# screen.keypad(False)
# curses.echo()
# curses.endwin()