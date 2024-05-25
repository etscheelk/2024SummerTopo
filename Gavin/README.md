## Gavin's Test Files
I know there's not a lot here. Right now, it just has some basic things with the network and topological libraries.

#### Files
The files in my folder are as follows:
- `concept_network.ipynb`: A notebook that breaks down how I went about creating a concept network. Still needs to have the normalized year (instead of actual year) and be ported into a function in a `.py` folder so it can be called easily in other documents in other analysis.
- `ripserer_exs.ipynb`: An attempt at using the Ripserer library to compute homology and find representative cycles.


#### To Do
I have a couple of things I want to work on:
- Create a "toy" dataset. Right now, the concept network to play with is massive, so any testing with it takes a long time. I want to create a smaller dataset that's somewhat representative of the features of the rest of the network so I can actually test things (ex persistence) without it being a difficult computational effort.
- Explore the concepts more to understand where errors lie. A couple errors I've seen are in edges between
    - "adaptive linear quadratic control" and "adaptive lq control"
    - "/ ms community" and "ms community"
- Explore OATpy more using other datasets, sparse matrices, and, if possible, networkx graphs (I've seen some functions take it as an input, but IDK what it does with it). I want to get an idea of how I can work with it and what I should be trying to use it for.
