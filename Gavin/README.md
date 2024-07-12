## Gavin's Test Files
My stuff. I tried to stay somewhat organized, so you can look below to see what each folder/file has

### Files
The files in my folder are as follows:
- `playtime`: A folder with various files that use the data and libraries. Should serve as an example for different functions, libraries, and things possible with our data
    - `concept_network.ipynb`: A notebook that breaks down how I went about creating a concept network. Still needs to have the normalized year (instead of actual year) and be ported into a function in a `.py` folder so it can be called easily in other documents in other analysis.
    - `ripserer_exs.ipynb`: An attempt at using the Ripserer library to compute homology and find representative cycles.
    - `oat_exs.ipynb`: An attempt at using the OATpy library to compute homology and representative cycles on various datasets.
    - `library_comparisons.ipynb`: A comparison of the solution time and results from different TDA libraries.
    - `my_first_cycle_rep.ipynb`: Solve for a cycle rep and create some visualizations of it.
    - `random_complexes.ipynb`: Create a handful of random graphs
    - `pmi_networks.ipynb`: Exploratory analysis of what PMI-based threasholding would mean for the network
    - `edge_weight_shuffle.ipynb`: Based on Adam's suggestion, we shuffle around the edge weights to different edges and see how the homology changes. Find that it creates a very different structure.
    - `goodlp_vs_gurobi.ipynb`: Comparisons of the results for OAT cycle optimizations between GoodLp and Gurobi. Finds that Gurobi is significantly faster (expected) and initially finds the 0.5 issues.
    - `optimizers_exs.ipynb`: Example usages of my `optimizers.py` utils classes. Shows how to optimize cycles and bounding chains using the functions.
    - `optimizing_a_bounding_chain.ipynb`: Explanation of the LP and methods used to optimize bounding chains
    - `optimizing_a_cycle_rep.ipynb`: Explanation of the LP and methods used to optimize cycle reps. This is technically wrong, and doesn't properly filter everything you should in the LP.
    - `optimizing_triangle_loss.ipynb`: Triangle loss optimizations to find cycle reps and bounding chains. This hasn't been implemented anywhere else (yet).
- `questions`: Files created to highlight confusing things I find in libraries/datasets.
    - `articles_dataset_questions.ipynb`: A file that highlights the repeated articles and weird IDs in the article file. For Russ.
    - `oat_3d_rep_cycles.ipynb`: A file that points out some confusions I have with the representative cycles returned by OATpy around 3D voids.
    - `ethan_help`: A folder with the initial question about saving the `FactoredBoundryMatrixVR` object. Idk what he's done to it at this point.
- `testing`: Test files. This folder is a mess, and you shouldn't look at it. Also, it's in the `.gitignore`, so it likely won't show up.
- `utils`: `.py` files with useful functions meant to be called in other files
    - `make_network_v1.py`: Functions that help to filter and create the network. Has since been updated
    - `samples.py`: Functions that create different samples of the network for testing various functions.
    - `compare_barcodes.py`: Functions that help compare barcodes from different libraries.
    - `random_complexes.py`: Functions for generating random simplicial complexes.
    - `make_network.py`: Functions that help to filter and create the network. Now allows filtering by edge characteristics
    - `optimizers.py`: Classes that can help with optimization. Should be faster than OAT when optimizing a bunch of cycles.
