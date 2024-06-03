## Gavin's Test Files
I know there's not a lot here. Right now, it just has some basic things with the network and topological libraries.

#### Files
The files in my folder are as follows:
- `playtime`: A folder with various files that use the data and libraries. Should serve as an example for what these can do.
    - `concept_network.ipynb`: A notebook that breaks down how I went about creating a concept network. Still needs to have the normalized year (instead of actual year) and be ported into a function in a `.py` folder so it can be called easily in other documents in other analysis.
    - `ripserer_exs.ipynb`: An attempt at using the Ripserer library to compute homology and find representative cycles.
    - `oat_exs.ipynb`: An attempt at using the OATpy library to compute homology and representative cycles on various datasets.
- `questions`: Files created to highlight confusing things I find in libraries/datasets.
    - `articles_dataset_questions.ipynb`: A file that highlights the repeated articles and weird IDs in the article file. For Russ.
    - `oat_3d_rep_cycles.ipynb`: A file that points out some confusions I have with the representative cycles returned by OATpy around 3D voids.
- `testing`: Test files. This folder is a mess and you shouldn't look at it. Also, it's in the `.gitignore` so it likely won't show up.
- `utils`: `.py` files with useful functions meant to be called in other files
    - `make_network.py`: Functions that help to filter and create the network.
    - `samples.py`: Functions that create different samples of the network for testing various functions.

#### To Do
I have a couple of things I want to work on:
- Create a "toy" dataset. Right now, the concept network to play with is massive, so any testing with it takes a long time. I want to create a smaller dataset that's somewhat representative of the features of the rest of the network so I can actually test things (ex persistence) without it being a difficult computational effort.
- Explore the concepts more to understand where errors lie. A couple errors I've seen are in edges between
    - "adaptive linear quadratic control" and "adaptive lq control"
    - "/ ms community" and "ms community"
- Explore OATpy more using other datasets, sparse matrices, and, if possible, networkx graphs (I've seen some functions take it as an input, but IDK what it does with it). I want to get an idea of how I can work with it and what I should be trying to use it for.
