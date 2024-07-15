'''
OAT has built in functions that find cycle reps and bounding chains, but these
don't always do exactly what we want and, I suspect, aren't as fast as they
could be. This is my attempt to solve for optimized cycles and bounding chains
in a more transparent, controlable, and possibly faster way.

All functions require a valid and accessable Gurobi License

It has the following classes:
    - JordanCycleOptimizer: A cycle optimzation problem that uses a Jordan cycle
    basis to setup the optimization problem. This creates a smaller constaint
    matrix that should be faster to solve the optimization problem, but takes
    longer to setup
    - BoundryHomologyCycleOptimizer: A cycle optimization problem that uses the
    cycle representatives from a homology calculation and the boundries from the
    boundry matrix. This creates LPs that take (slightly) longer to run, but
    should be able to be setup faster than the Jordan basis method
    - BoundingChainOptimizer: A optmiztion problem for bounding chains of cycles

All three clases have static methods that allow you to optimize a single cycle
and can be initialized to speed up the problem setup if you're solving more than
1 problem. 

TODO:
    - Create a better basis for the cycle problem
    - Area cost for Bounding Chains
    - Triangle loss problem
    - Test it
'''

# load some packages
from scipy import sparse
from enum import Enum
import gurobipy as gp
import pandas as pd
import numpy as np


class Cost(Enum):
    '''
    Possible cost functions for the optimization problems.

    Types:
        `UNIFORM`: Uniform weighting that will minimize the number of simplicies.
        May have issues with there being a lot of optimal chains
        `FILTRATION`: Weights based on the filtration. Therefore, it creates a
        chain with simplcicies that show up earlier
    '''
    UNIFORM = lambda chain: np.ones(len(chain))
    FILTRATION = lambda chain: chain['filtration'].to_numpy()


class LPType(Enum):
    '''
    Possible LP types for the optimization problems. Made for testing, in reality
    `POS_NEG` is faster for (almost?) all problems and should always be used

    Update: `MIN_VECTOR` might be faster for larger cycle rep problems

    Types
        `POS_NEG`: Uses x^+ and x^- to represent the values in the problem. Allows
        us to reprent an absolute value using x^+ + x^- and the actual solution as
        x^+ - x^-. Default for all problems
        `ABS_VALUE`: Uses a second y = |x| to solve the problem. Creates a lot of
        extra constraints, making it slower. Avalabile for all problems
        `MIN_VECTOR`: The method implemented in OAT. Solves y >= b + Ax and 
        y >= -b - Ax. The solution then is b+Ax. Avalabile for cycle rep problems
    '''
    POS_NEG = 1
    ABS_VALUE = 2
    MIN_VECTOR= 3


class Basis(Enum):
    '''
    A class for choosing the basis for a CycleOptimizer.

    A `JORDAN` basis uses a minimal Jordan basis for the cycles to solve the problem.
    The `CYCLE_BOUNDRY` basis uses all cycles and all boundries. The `JORDAN` basis
    will (usually) take longer to initialize but be faster to solve, while the
    `CYCLE_BOUNDRY` basis will be faster to initialize (since it uses built in OAT
    methods), but creates a larger constaint matrix that means the LP will be slower.
    If you want to optimize cycles after their death time, only the `CYCLE_BOUNDRY`
    basis will return the correct result, null, and the `JORDAN` basis will return a
    simplex within the cycle that has been filled in

    Types:
        `JORDAN`: Use a Jordan Basis for the constraint matrix
        `CYCLE_BOUNDRY`: Use all cycles and boundries as a basis for the constraint
        matrix. This will have singificant colinearity and is larger than it has to
        be
    '''
    JORDAN = 1
    CYCLE_BOUNDRY = 2


class ProblemType(Enum):
    '''
    A class for choosing the problem type to solve

    Essentailly, determines whether you use a strict or "inclusive" inequalty when
    deciding which cycles to include in the constraint matrix

    Types:
        `BEFORE_BIRTH`: Include all cycles that are born and die strictly before the
        cycle we optimize
        `AT_BIRTH`: Include all cycles that are born and die at the same time as or
        before the cycle we optimize
    '''
    BEFORE_BIRTH = lambda x, y: x < y
    AT_BIRTH = lambda x, y: x <= y


class CycleOptimizer:
    '''
    Class to optimize cycles.

    Uses either a jordan basis for the cycles or all cycles and boundries to create
    a constraint matrix and solves for a cycle that is the smallest possible subject
    to the constraint that its equal to the original cycle + any other ones

    Static Methods:
        `optimize_a_cycle`: Sets up an optimization problem and optimizes a cycle.
        Doesn't require the object to be initialized, since it does all of the problem
        setup from a factored object passed to it. This can be faster for optimizing
        a couple cycles at a time, but will be slower if you want more than a couple
        cycle reps
    
    Methods:
        `optimize_cycle`: Uses information stored within the optimizer to setup an
        optimization problem and find a minimal cycle rep
        `get_basis`: Return the basis the optimizer uses to optimize cycles

    Instance Variables:
        `cost` (Cost): The default cost to use in the LPs. Can be overwritten to
        make a different cost default
        `problem` (ProblemType): The default problem for the optimizer to solve.
        Can be overwritten to make a different solver default
        `integer` (bool): Whether to solve an integer or continuous Lp by default.
        Can be overwritten to make a different solver default
        `lp` (LPType): The default LP solution method for the optimizer to use.
        Can be overwritten to make a different solver default
        `simplex_indicies` (pd.DataFrame): The simplcies in the cycle we optmimze
        corresponding to the rows in the cycle basis
        `cycle_indicies` (pd.DataFrame): The columns of the cycle basis, each of
        which represents a different cycle (or simplex) in the simplicial complex
        `cycle_basis` (sparse.csr_matrix): The cycle basis sliced and used for
        optimization
        `gp_env` (gp.Env): Gruobi enviorment models are solved in
        `num_solved` (int): The number of bounding chains this opject has optimized
    '''

    # ----- Static Solver -----

    @staticmethod
    def optimize_a_cycle(birth_simplex: list[int],
                         factored: any, # should be a FactoredBoundryMatrixVR
                         filtration: float | None = None,
                         return_objective: bool = False,
                         cost: Cost = Cost.FILTRATION,
                         problem: ProblemType = ProblemType.AT_BIRTH,
                         integer: bool = False,
                         basis: Basis = Basis.JORDAN,
                         lp: LPType = LPType.POS_NEG
                         ) -> pd.DataFrame | tuple[pd.DataFrame, float]:
        '''
        Optimizes a cycle rep. This can be faster than instantiating the object for
        individual cycles, but will be slower if you want to optimize many cycles at
        once, since it can't resuse any of the setup work

        Args:
            `birth_simplex` (list[int]): The birth simplex for the cycle you want to
            optimize. Used to identify the cycle
            `factored` (FactoredBoundryMatrixVR): A factored matrix for the simplcial
            complex. Should have max_dim set at least 1 above the cycle dimension
            `filtration` (float | None): The filtration value to find a cycle at. If
            set to None, uses the birth filtration of the cycle. Default None.
            `return_objective` (bool): Whether or not to return the value of the
            optimized objective. If False, just returns the optimized cycle. If True,
            returns a tuple with the cycle and the objective
            `cost` (Cost): The cost weighting. If UNIFORM, we minimize the number of
            simplicies in the cycle. If FILTRATION, we minimize the total filtration
            of the cycle. Default FILTRATION
            `problem` (ProblemType): The filter applied to the cycle to decide whether
            they go in the LP. Default AT_BIRTH, which includes all cycles that are
            born and die at or before the cycle we optimize
            `integer` (bool): Whether to solve an integer or continous LP. Default
            False
            `basis` (Basis): The basis to use for the cycles. Default JORDAN, whcih
            creates a minimal cycle basis
            `lp` (LPType): The LP to solve to optimize the cycle. POS_NEG is
            fastest and, therefore,  default

        Returns:
            `optimal_cycle_rep` (pd.DataFrame): A dataframe representing the cycle.
            Has columns for "simplex", "filtration", and "coefficient"
            `obj` (float): The objective value of the cycle. Returned only if 
            `return_objective` is True.
        '''
        # setup
        cycle_dim = len(birth_simplex) - 1 # cycle dimension is always 1 less than the number of elements in the simplcicies
        cycle_rep = factored.jordan_basis_vector(birth_simplex) # the initial cycle rep to optimize
        if filtration is None: # define the filtration if None
            filtration = cycle_rep['filtration'].max()
        
        # simplicies we optimize over
        simplex_indicies = factored.indices_boundary_matrix()
        simplex_indicies = simplex_indicies[(simplex_indicies['simplex'].str.len() == cycle_dim+1) # keep simplicies of the same dimension
                                            & (problem(simplex_indicies['filtration'], filtration) # keep simplcicies born at or before the time we're looking for
                                               | simplex_indicies['simplex'].isin(cycle_rep['simplex'])) # keep simplicice in the cycle
                                            ].reset_index(drop=True) # well use the indexes to create the basis matrix
        
        # simplex -> index in everything map
        simplex_index_map = pd.DataFrame(simplex_indicies['simplex'].map(tuple) # make it a mapable datatype (lists cant be mapped)
                                         ).reset_index( # make 'index' a column, should be the index of the simplex in the matrix
                                         ).set_index('simplex') # index by simplex
        
        # fill in the 0 coefficeints in the cycle rep
        initial_cycle_rep = simplex_indicies.copy() # cycle rep should have all simplcicies in it (most will be 0)
        initial_cycle_rep['coefficient'] = 0. # defualt coefficeint is 0
        cycle_is = simplex_index_map.loc[cycle_rep['simplex'].map(tuple), 'index'].tolist() # where nonzero coefficeints are located
        initial_cycle_rep.loc[cycle_is, 'coefficient'] = cycle_rep['coefficient'].astype(float).tolist() # fill in nonzero coefficeints
        # need to turn everything into a list bc its a pandas array so the indexes either need to align (which they dont) or be a list

        # create the cycle basis to serve as a constraint
        match basis:
            # we want to use a jordan basis
            case Basis.JORDAN:
                # get cycles we want to find
                jordan_indicies = factored.jordan_block_indices()
                cycle_i = jordan_indicies.loc[jordan_indicies['birth simplex'].apply(tuple) == tuple(birth_simplex)].index[0] # index of the cycle
                death = max(jordan_indicies.loc[cycle_i, 'death filtration'], filtration)
                jordan_indicies = jordan_indicies[(jordan_indicies['dimension'] == cycle_dim) # keep cycles of the same dimension
                                                  & (problem(jordan_indicies['birth filtration'], filtration)) # keep cycles born at or before the time we're looking for
                                                  & (problem(jordan_indicies['death filtration'], death))
                                                  & (jordan_indicies.index != cycle_i)] # don't have the orginal cycle in the basis

                # create the basis
                cycles = [factored.jordan_basis_vector(b)[['simplex', 'coefficient']] for b in jordan_indicies['birth simplex']]
                cycle_basis = CycleOptimizer.__create_cycle_basis(cycles, simplex_index_map)

            # we want to use a cycle/boundry basis
            case Basis.CYCLE_BOUNDRY:
                # create the cycle basis
                homology = factored.homology(
                        return_cycle_representatives=True, # used to create the cycle basis
                        return_bounding_chains=False
                    )[['dimension', 'birth', 'birth simplex', 'death', 'cycle representative']] # we only care about these columns
                cycle_i = homology.loc[homology['birth simplex'].apply(tuple) == tuple(birth_simplex)].index[0] # index of the cycle
                death = max(homology.loc[cycle_i, 'death'], filtration)
                homology = homology[(homology['dimension'] == cycle_dim) # make basis of cycles in the right dimension
                                    & problem(homology['birth'], filtration) # basis of cycles born before the time were looking at
                                    & problem(homology['death'], death)
                                    & (homology.index != cycle_i)] # dont have cycle of interest in basis
                cycle_basis = CycleOptimizer.__create_cycle_basis(homology['cycle representative'], simplex_index_map)

                # create the boundry basis
                simplex_indicies = factored.indices_boundary_matrix() # we could prolly reuse this from before, but its fast so idc
                cycle_dim_simplicies = simplex_indicies[(simplex_indicies['simplex'].str.len() == cycle_dim+1) # keep simplicies of the same dimension
                                                        & (problem(simplex_indicies['filtration'], filtration) # keep simplcicies born at or before the time we're looking for
                                                           # don't need death filter since filtration and death times are the same for simplicies
                                                           | simplex_indicies['simplex'].isin(cycle_rep['simplex'])) # keep simplicice in the cycle
                                                        ] # well use the indexes to create the boundry matrix
                higher_dim_simplicies = simplex_indicies[(simplex_indicies['simplex'].str.len() == cycle_dim+2) # keep simplicies of the same dimension
                                                         & problem(simplex_indicies['filtration'], filtration) # keep simplcicies born at or before the time we're looking for
                                                         ] # well use the indexes to create the boundry matrix
                boundry_matrix = factored.boundary_matrix(
                    ).astype(float # everyhting likes floats more
                    )[cycle_dim_simplicies.index, :][:, higher_dim_simplicies.index] # filter to the right indicies

                # combine them into one matrix
                cycle_basis = sparse.hstack((cycle_basis, boundry_matrix))

        # make sure theres something to optimize
        if cycle_basis.shape[1] == 0:
            optimal_cycle_rep = CycleOptimizer.__collect_results(initial_cycle_rep, initial_cycle_rep['coefficient'])

            # return
            if return_objective:
                return optimal_cycle_rep, np.abs(cost(optimal_cycle_rep) * optimal_cycle_rep['coefficient']).sum()
            return optimal_cycle_rep

        # get cost
        cost = cost(initial_cycle_rep)

        # setup problem and solve
        coeffs, obj = CycleOptimizer.__optimize(
                initial_cycle_rep['coefficient'].to_numpy(),
                cycle_basis,
                cost,
                integer,
                lp
            )
        
        # get solution
        optimal_cycle_rep = CycleOptimizer.__collect_results(initial_cycle_rep, coeffs)

        # solve the model
        if return_objective:
            return optimal_cycle_rep, obj
        return optimal_cycle_rep


    # ----- Methods -----

    def __init__(self,
                 factored: any, # FactoredBoundryMatrixVR
                 cycle_dim: int,
                 cost: Cost = Cost.FILTRATION,
                 problem: ProblemType = ProblemType.AT_BIRTH,
                 integer: bool = False,
                 basis: Basis = Basis.JORDAN,
                 lp: LPType = LPType.POS_NEG,
                 supress_gurobi: bool = False
                 ) -> None:
        '''
        Intantiate a CycleOptimizer. Does all of the setup, which makes optimizing
        cycles later much faster.

        Args:
            `factored` (FactoredBoundryMatrixVR): The factored matrix for the cycles.
            Should have dimension cycle_dim+1 at least
            `cycle_dim` (int: Dimension of the cycles we want cycle reps for
            `cost` (Cost): The cost to use to optimize bounding chains. Can be either
            FILTRATION or UNIFORM. Default FILTRATION
            `problem` (ProblemType): The default filter applied to the cycle to decide
            whether they go in the LP. Default AT_BIRTH, which includes all cycles that
            are born and die at or before the cycle we optimize
            `integer` (bool): Whether to solve an integer or continous LP by default.
            Default False
            `basis` (Basis): The basis to use for the cycles. Default JORDAN, whcih
            creates a minimal cycle basis
            `lp` (LPType): The default LP for the optimizer to use to solve
            for optimal bounding chains. Can be POS_NEG, ABS_VALUE, or MIN_VEC, though
            POS_NEG is faster and suggested. Default POS_NEG.
            `supress_gurobi` (bool): Whether to supress Gurobi output or not. Default
            False
        '''
        # define default problem/cost
        self.cost = cost
        self.problem = problem
        self.integer = integer
        self.__basis = basis
        self.lp = lp
        self.num_solved = 0

        # simplicies we optimize over
        self.simplex_indicies = factored.indices_boundary_matrix()
        self.simplex_indicies['filtration'] = self.simplex_indicies['filtration'].astype(float)
        self.simplex_indicies = self.simplex_indicies[(self.simplex_indicies['simplex'].str.len() == cycle_dim+1) # keep simplicies of the same dimension
                                                      ].reset_index(drop=True) # well use the indexes to create the basis matrix
        
        # simplex -> row in everything map
        simplex_index_map = pd.DataFrame(self.simplex_indicies['simplex'].map(tuple) # make it a mapable datatype (lists cant be mapped)
                                         ).reset_index( # make 'index' a column, should be the index of the simplex in the matrix
                                         ).set_index('simplex') # index by simplex
        
        # create the cycle basis to serve as a constraint
        match basis:
            # jordan basis
            case Basis.JORDAN:
                # indicies for everything
                jordan_indicies = factored.jordan_block_indices()
                jordan_indicies = jordan_indicies[jordan_indicies['dimension'] == cycle_dim # keep cycles of the correct dimension
                                                  ].reset_index(drop=True) # well use the indicies to slice the basis matrix
                
                # create the cycle basis
                cycles = [factored.jordan_basis_vector(b)[['simplex', 'coefficient']] for b in jordan_indicies['birth simplex']]
                self.cycle_basis = CycleOptimizer.__create_cycle_basis(cycles, simplex_index_map)
                
                # formatting
                jordan_indicies['birth simplex'] = jordan_indicies['birth simplex'].apply(tuple) # hashable
                self.cycle_indicies = jordan_indicies[['birth simplex', 'birth filtration', 'death filtration']].rename(columns={ # save space
                        'birth filtration': 'birth',
                        'death filtration': 'death'
                    })

            # cycle boundry basis
            case Basis.CYCLE_BOUNDRY:
                # create the cycle basis
                homology = factored.homology(
                        return_cycle_representatives=True, # used to create the cycle basis
                        return_bounding_chains=False
                    )[['dimension', 'birth', 'birth simplex', 'death', 'cycle representative']] # we only care about these columns
                homology = homology[homology['dimension'] == cycle_dim # make basis of cycles in the right dimension
                                    ].reset_index(drop=True) # well use the indicies to slice the basis matrix
                cycle_basis = CycleOptimizer.__create_cycle_basis(homology['cycle representative'], simplex_index_map)

                # create the boundry basis
                simplex_indicies = factored.indices_boundary_matrix() # we could prolly reuse this from before, but its fast so idc
                cycle_dim_simplicies = simplex_indicies[simplex_indicies['simplex'].str.len() == cycle_dim+1] # keep simplicies of the same dimension
                higher_dim_simplicies = simplex_indicies[simplex_indicies['simplex'].str.len() == cycle_dim+2] # keep simplicies of the same dimension
                boundry_matrix = factored.boundary_matrix(
                    ).astype(float # everyhting likes floats more
                    )[cycle_dim_simplicies.index, :][:, higher_dim_simplicies.index] # filter to the right indicies
                
                # combine them into one matrix
                self.cycle_basis = sparse.hstack((cycle_basis, boundry_matrix))

                # formatting
                self.cycle_indicies = pd.concat([ # matrix is indexed by homology on top of the boundries
                        homology.assign(**{
                                'birth simplex': lambda h: h['birth simplex'].apply(tuple) # make this hashable
                            })[['birth simplex', 'birth', 'death']], # keep relevant columns
                        higher_dim_simplicies.assign(simplex=None  # the simplicices are too high dimension to matter
                            ).assign(death=higher_dim_simplicies['filtration'] # cycle "dies" at the same time it's born
                            ).rename(columns={
                                'simplex': 'birth simplex',
                                'filtration': 'birth'
                            })
                    ]).reset_index(drop=True) # cycle basis is indexed starting at 0

        # create the model enviorment
        self.gp_env = gp.Env()
        if supress_gurobi:
            self.gp_env.setParam('OutputFlag', 0)


    def optimize_cycle(self,
                       birth_simplex: pd.DataFrame,
                       filtration: float | None = None,
                       return_objective: bool = False,
                       cost: Cost | None = None,
                       problem: ProblemType | None = None,
                       integer: bool | None = None,
                       lp: LPType | None = None
                       ) -> pd.DataFrame | tuple[pd.DataFrame, float]:
        '''
        Optimizes a cycle. Uses the setup from the object to speed up the process and
        solve the LP faster

        Args:
            `birth_simplex` (list[int]): The birth simplex for the cycle you want to
            optimize. Used to identify the cycle
            `filtration` (float | None): The filtration value to find a cycle at. If
            set to None, uses the birth filtration of the cycle. Default None.
            `return_objective` (bool): Whether or not to return the value of the
            optimized objective. If False, just returns the optimized cycle. If True,
            returns a tuple with the cycle and the objective
            `cost` (Cost | None): The cost to use. Can be FILTRATION or UNIFORM. If
            None, uses the deafult for the optimizer. Default None
            `problem` (ProblemType | None):  The default filter applied to the cycle
            to decide whether they go in the LP. If None, uses the default for the
            optimizer. Default None.
            `integer` (bool | None): Whether to solve an integer or continous LP. If
            None, uses the default for the optimizer. Default None.
            `lp` (LPType | None): The LP to solve to optimize the cycle. If
            None, uses the default for the optimizer. Default None

        Returns:
            `optimal_cycle_rep` (pd.DataFrame): A dataframe representing the cycle.
            Has columns for "simplex", "filtration", and "coefficient"
            `obj` (float): The objective value of the cycle. Returned only if 
            `return_objective` is True.
        '''
        # set problem and cost
        cost = self.cost if cost is None else cost
        problem = self.problem if problem is None else problem
        integer = self.integer if integer is None else integer
        lp = self.lp if lp is None else lp
        self.num_solved += 1

        # where the cycle is
        cycle_i = self.cycle_indicies[self.cycle_indicies['birth simplex'] == tuple(birth_simplex)].index[0] # get index for the cycle 

        # figure out filtration
        if filtration is None:
            filtration = self.cycle_indicies.loc[cycle_i, 'birth']
        death = max(self.cycle_indicies.loc[cycle_i, 'death'], filtration)

        # initial cycle rep
        initial_coeffs = np.reshape(self.cycle_basis[self.simplex_indicies.index, cycle_i].toarray(), -1)

        # simplcicies to include
        simplex_indicies = self.simplex_indicies[problem(self.simplex_indicies['filtration'], filtration)
                                                 | (initial_coeffs != 0)] # include everything born before the filtration
        initial_coeffs = initial_coeffs[simplex_indicies.index]

        # cycle basis 
        cycle_indicies = self.cycle_indicies[problem(self.cycle_indicies['birth'], filtration) # keep cycles born at or before the time we're looking for
                                             & problem(self.cycle_indicies['death'], death)
                                             & (self.cycle_indicies.index != cycle_i)] # don't have orginal cycle in the basis
        
        # make sure theres a problem to solve
        if len(cycle_indicies) == 0:
            optimal_cycle_rep = CycleOptimizer.__collect_results(simplex_indicies, initial_coeffs)

            # return 
            if return_objective:
                return optimal_cycle_rep, np.abs(cost(optimal_cycle_rep) * optimal_cycle_rep['coefficient']).sum()
            return optimal_cycle_rep
        
        # get cost
        cost = cost(simplex_indicies)

        # cycle basis pt 2
        cycle_basis = self.cycle_basis[simplex_indicies.index, :][:, cycle_indicies.index]

        # setup problem and solve
        coeffs, obj = CycleOptimizer.__optimize(
                initial_coeffs,
                cycle_basis,
                cost,
                integer,
                lp,
                env=self.gp_env
            )
        
        # get solution
        optimal_cycle_rep = CycleOptimizer.__collect_results(simplex_indicies, coeffs)

        # solve the model
        if return_objective:
            return optimal_cycle_rep, obj
        return optimal_cycle_rep
    

    def close(self):
        '''
        Call this before the program ends. Closes the optimizer
        '''
        self.gp_env.close()


    def get_basis(self):
        '''
        Returns the basis used to optimize cycles
        '''
        return self.__basis


    # ----- Helper Functions ------

    @staticmethod
    def __create_cycle_basis(cycles, simplex_index_map):
        '''
        Maps cycles into a cycle basis choosing indicies based on the simplex_index_map
        '''        
        # create the cycle basis matrix
        cycle_basis = sparse.csr_matrix((len(simplex_index_map), len(cycles))) # create a cycle basis matrix

        # fill in the cycle basis matrix
        for i, c in enumerate(cycles):
            c_i = simplex_index_map.loc[c['simplex'].map(tuple), 'index'] # get indcices of simplcicies in cycle
            cycle_basis[c_i, i] = c['coefficient'].astype(float) # put coefficients into cycle basis

        return cycle_basis
            
    @staticmethod
    def __optimize(cycle_rep, cycle_basis, cost, integer, lp, env=None):
        '''
        Decides which problem to use and passes the model params to that
        '''
        # initialize model
        model = gp.Model(env=env)

        # solve problem
        match lp:
            case LPType.POS_NEG:
                coeffs = CycleOptimizer.__optimize_pos_neg(model, cycle_rep, cycle_basis, cost, integer)
            case LPType.ABS_VALUE:
                coeffs = CycleOptimizer.__optimize_abs_value(model, cycle_rep, cycle_basis, cost, integer)
            case LPType.MIN_VECTOR:
                coeffs = CycleOptimizer.__optimize_min_vactor(model, cycle_rep, cycle_basis, cost, integer)
        obj = model.getObjective().getValue() # objective value

        # close model
        model.close()

        return coeffs, obj

    @staticmethod
    def __optimize_pos_neg(model, cycle_rep, cycle_basis, cost, integer):
        '''
        Optimize an LP solving

        min         c^T (x^+ + x^-)
        subject to  x^+ - x^- = b + Az
        '''
        # decide whether to solve integer or continuous program
        vtype = gp.GRB.INTEGER if integer else gp.GRB.CONTINUOUS

        # free (decision) variables
        pos_coeffs = model.addMVar((len(cycle_rep)), vtype=vtype) # the positive coefficeints
        neg_coeffs = model.addMVar((len(cycle_rep)), vtype=vtype) # the negative coefficeints
        cycles = model.addMVar((cycle_basis.shape[1]), lb=-gp.GRB.INFINITY) # the cycles that we add together

        # constraints
        model.addConstr(pos_coeffs - neg_coeffs == cycle_rep + cycle_basis @ cycles) # cycle surrounds what we care about

        # objective
        model.setObjective(gp.quicksum(cost * (pos_coeffs + neg_coeffs)))

        # solve
        model.optimize()

        return pos_coeffs.X - neg_coeffs.X
    
    @staticmethod
    def __optimize_abs_value(model, cycle_rep, cycle_basis, cost, integer):
        '''
        Optimize an LP solving

        min         c^T y
        subject to  y = |x|
                    x = b + Az
        '''
        # decide whether to solve integer or continuous program
        vtype = gp.GRB.INTEGER if integer else gp.GRB.CONTINUOUS

        # free (decision) variables
        coeffs = model.addMVar((len(cycle_rep)), lb=-gp.GRB.INFINITY, vtype=vtype) # the coefficeints (what we return)
        abs_coeffs = model.addMVar((len(cycle_rep)), vtype=vtype) # absolute value of the coefficeints (cost function)
        cycles = model.addMVar((cycle_basis.shape[1]), lb=-gp.GRB.INFINITY, vtype=vtype) # the cycles that we add together

        # constraints
        model.addConstrs((abs_coeffs[i] == gp.abs_(coeffs[i]) for i in range(len(cycle_rep)))) # absolute value condition
        model.addConstr(coeffs == cycle_rep + cycle_basis @ cycles) # cycle surrounds what we care about

        # objective
        model.setObjective(gp.quicksum(cost * abs_coeffs))

        # solve
        model.optimize()

        return coeffs.X
    
    @staticmethod
    def __optimize_min_vactor(model, cycle_rep, cycle_basis, cost, integer):
        '''
        Optimize an LP solving

        min         c^T y
        subject to  y >= b + Az
                    y >= -b - Az
        '''
        # decide whether to solve integer or continuous program
        vtype = gp.GRB.INTEGER if integer else gp.GRB.CONTINUOUS

        # free (decision) variables
        abs_coeffs = model.addMVar((len(cycle_rep)), vtype=vtype) # absolute value of the coefficeints (cost function)
        cycles = model.addMVar((cycle_basis.shape[1]), lb=-gp.GRB.INFINITY, vtype=vtype) # the cycles that we add together

        # constraints
        model.addConstr(abs_coeffs >= cycle_rep + cycle_basis @ cycles) # cycle surrounds what we care about
        model.addConstr(abs_coeffs >= -cycle_rep - cycle_basis @ cycles) # cycle surrounds what we care about

        # objective
        model.setObjective(gp.quicksum(cost * abs_coeffs))

        # solve
        model.optimize()

        return cycle_rep + cycle_basis @ cycles.X
    
    @staticmethod
    def __collect_results(simplex_indicies, coeffs):
        '''
        Take the results of the optimization problem and turn it into a returnable cycle rep
        '''
        optimal_cycle_rep = simplex_indicies.assign(coefficient=coeffs) # add eoffcients
        optimal_cycle_rep = optimal_cycle_rep[optimal_cycle_rep['coefficient'] != 0].reset_index(drop=True) # keep only nonzero entries

        return optimal_cycle_rep[['simplex', 'filtration', 'coefficient']]



class BoundingChainOptimizer:
    '''
    Class to optimize bounding chains for inputted cycle reps

    Uses the boundry as the constraint matrix and solves for the minimal bounding
    chain such that the boundry is the inputted cycle

    Static Methods:
        `optimize_a_bounding_chain`: Optimizes the bounding chain of an inputted
        cycle. A good option if you only want to optimize a couple cycles, but
        since it does the entire setup every time, it's much slower than
        instantiating the object for doing a large number of cycle reps
    
    Methods:
        `optimize_bounding_chain`: Optimized the bounding chain of an inputted
        cycle using variables setup at initialization. Much faster than the static
        method, but requires the object to be setup
    
    Instance Variables:
        `cost` (Cost): The default cost to use in the LPs. Can be overwritten to
        make a different cost default
        `integer` (bool): Whether to solve an integer or continuous Lp by default.
        Can be overwritten to make a different solver default
        `lp` (LPType): The default LP solution method for the optimizer to use. Can
        be overwritten to make a different solver default
        `cycle_rep` (pd.DataFrame): The simplcices and their filtration parameters
        that can be in the cycles
        `bounding_chain` (pd.DataFrame): The simplcices and their filtration
        parameters that can be in the bounding chains
        `boundry_matrix` (sparse.csr_matrix): The boundry matrix for optimzation.
        The rows correspond to their index in `cycle_rep` and columns to their
        index in `bounding_chain`
        `gp_env` (gp.Env): Gruobi enviorment models are solved in
        `num_solved` (int): The number of bounding chains this opject has optimized
    '''

    # ----- Static Solver -----

    @staticmethod
    def optimize_a_bounding_chain(cycle: pd.DataFrame,
                                  factored: any, # should be a FactoredBoundryMatrixVR
                                  death: float = np.inf, # is there a way to solve for this from the cycle?
                                  return_objective: bool = False,
                                  cost: Cost = Cost.FILTRATION,
                                  integer: bool = False,
                                  lp: LPType = LPType.POS_NEG
                                  ) -> pd.DataFrame | tuple[pd.DataFrame, float]:
        '''
        Optimizes the bounding chain for a given cycle. If you're trying to do
        many cycles, this will be much slower than instantiating the object since
        it does all the setup every time

        Args:
            `cycle` (pd.Dataframe): Dataframe for the cycle you want to optimize.
            Should have columns for "simplex", "filtration", and "coefficient"
            `factored` (FactoredBoundryMatrixVR): The factored matrix to use. Should
            have dimension cycle_dim+1 at least
            `death` (float): The death time of the cycle. Searches for a bounding
            chain before this death. Default inf, meaning it searches for a bounding
            chain that may or may not exist when the cycle dies
            `return_objective` (bool): Whether to return the objective solution
            or not. If True, returns a tuple ('optimal_bounding_chain', 'obj'). If
            False, just returns 'optimal_bounding_chain'. Default False
            `cost` (Cost): The cost to use. Can be FILTRATION or UNIFORM. If None, 
            uses the deafult for the optimizer. Default None
            `integer` (bool): Whether to solve an integer or continous LP. Default
            False
            `lp` (LPType | None): The LP to solve. Can be POS_NEG or ABS_VALUE. If
            None, uses the default for the optimizer. Default None

        Returns:
            `optimal_bounding_chain` (pd.DataFrame): Dataframe with the optimal
            bounding chain for the cycle. Has columns "simplex", "filtration",
            and "coefficient".
            `obj` (float): Returned if `return_objective` is True. The optimal
            cost found
        '''
        # get basic information about the cycle
        cycle_dim = len(cycle.loc[0, 'simplex']) - 1 # dimension is 1 less than the length of the simplices
        simplex_indicies = factored.indices_boundary_matrix()
        simplex_indicies['filtration'] = simplex_indicies['filtration'].astype(float)
        # everythgin likes floats more

        # simplex keys, index will be location in boundry_matrix
        cycle_rep = simplex_indicies[(simplex_indicies['simplex'].str.len() == cycle_dim+1) # boundry (cycle) simplicies should be the same dimension as the cycle
                                     & (simplex_indicies['filtration'] <= death)].copy() # everything should exist before the death time
        cycle_rep = BoundingChainOptimizer.__get_cycle_rep_with_zeros(cycle, cycle_rep) # add 0 coefficeints
        bounding_chain = simplex_indicies[(simplex_indicies['simplex'].str.len() == cycle_dim+2) # bounding chain simplicies hsould have 1 higher dimension
                                          & (simplex_indicies['filtration'] <= death)].copy()
        
        # boundry matrix
        boundry_matrix = factored.boundary_matrix().astype(float)[cycle_rep.index, :][:, bounding_chain.index]

        # get cost
        cost = cost(bounding_chain)

        # figure out which problem to use and solve
        coeffs, obj = BoundingChainOptimizer.__optimize(
                cycle_rep['coefficient'].to_numpy(),
                boundry_matrix,
                cost,
                integer,
                lp
            )
                
        # get solution
        optimal_bounding_chain = BoundingChainOptimizer.__collect_results(bounding_chain, coeffs)

        # solve the model
        if return_objective:
            return optimal_bounding_chain, obj
        return optimal_bounding_chain    


    # ----- Methods -----

    def __init__(self,
                 factored: any, # FactoredBoundryMatrixVR
                 cycle_dim: int,
                 cost: Cost = Cost.FILTRATION,
                 integer: bool = False,
                 lp: LPType = LPType.POS_NEG,
                 supress_gurobi: bool = False
                 ) -> None:
        '''
        Setups up the optimizer. Does most of the work so that future optimizations
        are faster.

        Args:
            `factored` (FactoredBoundryMatrixVR): The factored matrix for the cycles.
            Should have dimension cycle_dim+1 at least
            `cycle_dim` (int: Dimension of the cycles we want bounding chains for
            `cost` (Cost): The cost to use to optimize bounding chains. Can be either
            FILTRATION or UNIFORM. Default FILTRATION
            `integer` (bool): Whether to solve an integer or continous LP by default.
            Default False
            `lp` (LPType): The default LP for the optimizer to use to solve
            for optimal bounding chains. Can be either POS_NEG or ABS_VALUE, though
            POS_NEG is faster and suggested. Default POS_NEG.
            `supress_gurobi` (bool): Whether to supress Gurobi output or not. Default
            False
        '''
        # define default problem/cost
        self.cost = cost
        self.integer = integer
        self.lp = lp
        self.num_solved = 0

        # simplcices we use
        simplex_indicies = factored.indices_boundary_matrix()
        simplex_indicies['filtration'] = simplex_indicies['filtration'].astype(float)
        self.cycle_rep = simplex_indicies[simplex_indicies['simplex'].str.len() == cycle_dim+1]
        self.bounding_chain = simplex_indicies[simplex_indicies['simplex'].str.len() == cycle_dim+2]

        # boundry matrix
        self.boundry_matrix = factored.boundary_matrix().astype(float)[self.cycle_rep.index, :][:, self.bounding_chain.index]
        # this will be filtered down in individual solvers
        self.cycle_rep = self.cycle_rep.reset_index(drop=True) # boundry matrix inidicies start at 0 now, not 
        self.bounding_chain = self.bounding_chain.reset_index(drop=True)

        # create the model enviorment
        self.gp_env = gp.Env()
        if supress_gurobi:
            self.gp_env.setParam('OutputFlag', 0)

    
    def optimize_bounding_chain(self,
                                cycle: pd.DataFrame,
                                death: float = np.inf,
                                return_objective: bool = False,
                                cost: Cost | None = None,
                                integer: bool | None = None,
                                lp: LPType | None = None
                                ) -> pd.DataFrame | tuple[pd.DataFrame, float]:
        '''
        Optimizes the bounding chain for a given cycle

        Args:
            `cycle` (pd.Dataframe): Dataframe for the cycle you want to optimize.
            Should have columns for "simplex", "filtration", and "coefficient"
            `death` (float): The death time of the cycle. Searches for a bounding
            chain before this death. Default inf, meaning it searches for a bounding
            chain that may or may not exist when the cycle dies
            `return_objective` (bool): Whether to return the objective solution
            or not. If True, returns a tuple ('optimal_bounding_chain', 'obj'). If
            False, just returns 'optimal_bounding_chain'. Default False
            `cost` (Cost): The cost to use. Can be FILTRATION or UNIFORM. If None, 
            uses the deafult for the optimizer. Default None
            `integer` (bool | None): Whether to solve an integer or continous LP. If
            None, uses the default for the optimizer. Default None.
            `lp` (LPType | None): The LP to solve. Can be POS_NEG or
            ABS_VALUE. If None, uses the default for the optimizer. Default None

        Returns:
            `optimal_bounding_chain` (pd.DataFrame): Dataframe with the optimal
            bounding chain for the cycle. Has columns "simplex", "filtration",
            and "coefficient".
            `obj` (float): Returned if `return_objective` is True. The optimal
            cost found
        '''
        # set problem and cost
        cost = self.cost if cost is None else cost
        integer = self.integer if integer is None else integer
        lp = self.lp if lp is None else lp
        self.num_solved += 1

        # cycle rep and bounding chain indicies
        cycle_rep = self.cycle_rep[self.cycle_rep['filtration'] <= death].copy()
        cycle_rep = BoundingChainOptimizer.__get_cycle_rep_with_zeros(cycle, cycle_rep)
        bounding_chain = self.bounding_chain[self.bounding_chain['filtration'] <= death]

        # get boundry matrix
        boundry_matrix = self.boundry_matrix[cycle_rep.index, :][:, bounding_chain.index]

        # get cost
        cost = cost(bounding_chain)

        # figure out which problem to use and solve
        coeffs, obj = BoundingChainOptimizer.__optimize(
                cycle_rep['coefficient'].to_numpy(),
                boundry_matrix,
                cost,
                integer,
                lp,
                env=self.gp_env
            )
                
        # get solution
        optimal_bounding_chain = BoundingChainOptimizer.__collect_results(bounding_chain, coeffs)

        # solve the model
        if return_objective:
            return optimal_bounding_chain, obj
        return optimal_bounding_chain
    

    def close(self):
        '''
        Call this before the program ends. Closes the optimizer
        '''
        self.gp_env.close()


    # ----- Helper Functions ------

    @staticmethod
    def __get_cycle_rep_with_zeros(cycle, simplex_indicies):
        '''
        Adds coefficeints to the simplcies to get the cycle rep with 0s
        '''
        simplex_indicies['coefficient'] = 0. # set default coefficent to 0
        cycle_i = pd.DataFrame(
                simplex_indicies['simplex'].map(tuple) # need hashable thing to index on
            ).reset_index( # make index a column
            ).set_index('simplex' # search by simplex
            ).loc[cycle['simplex'].apply(tuple), 'index'].tolist() # find the cycle
        simplex_indicies.loc[cycle_i, 'coefficient'] = cycle['coefficient'].tolist()

        return simplex_indicies
    
    @staticmethod
    def __optimize(cycle_rep, boundry_matrix, cost, integer, lp, env=None):
        '''
        Figures out which lp to solve as passes it to the right solver
        '''
        # create the model
        model = gp.Model(env=env)

        # solve the LP
        match lp:
            case LPType.POS_NEG:
                coeffs = BoundingChainOptimizer.__optimize_pos_neg(model, cycle_rep, boundry_matrix, cost, integer)
            case LPType.ABS_VALUE:
                coeffs = BoundingChainOptimizer.__optimize_abs_value(model, cycle_rep, boundry_matrix, cost, integer)
            case LPType.MIN_VECTOR:
                raise NotImplementedError('Use only POS_NEG or ABS_VALUE for bounding chains')
            
        obj = model.getObjective().getValue() # objective value

        # close model
        model.close()

        return coeffs, obj
    
    @staticmethod
    def __optimize_pos_neg(model, cycle_rep, boundry_matrix, cost, integer):
        '''
        Optimize an LP solving

        min         c^T (x^+ + x^-)
        subject to  b = A (x^+ - x^-)
        '''
        # decide whether to solve integer or continuous program
        vtype = gp.GRB.INTEGER if integer else gp.GRB.CONTINUOUS

        # free (decision) variables
        pos_coeffs = model.addMVar((boundry_matrix.shape[1])) # the positive coefficeints
        neg_coeffs = model.addMVar((boundry_matrix.shape[1])) # the negative coefficeints

        # constraints
        model.addConstr(boundry_matrix @ (pos_coeffs-neg_coeffs) == cycle_rep) # cycle surrounds what we care about

        # objective
        model.setObjective(gp.quicksum(cost * (pos_coeffs + neg_coeffs)))

        # solve
        model.optimize()

        return pos_coeffs.X - neg_coeffs.X

    @staticmethod
    def __optimize_abs_value(model, cycle_rep, boundry_matrix, cost, integer):
        '''
        Optimize an LP solving

        min         c^T y
        subject to  y = |x|
                    b = A x
        '''
        # decide whether to solve integer or continuous program
        vtype = gp.GRB.INTEGER if integer else gp.GRB.CONTINUOUS

        # free (decision) variables
        coeffs = model.addMVar((boundry_matrix.shape[1]), lb=-gp.GRB.INFINITY) # the positive coefficeints
        abs_coeffs = model.addMVar((boundry_matrix.shape[1])) # the negative coefficeints

        # constraints
        model.addConstrs((abs_coeffs[i] == gp.abs_(coeffs[i]) for i in range(boundry_matrix.shape[1]))) # absolute value condition
        model.addConstr(boundry_matrix @ coeffs == cycle_rep) # cycle surrounds what we care about

        # objective
        model.setObjective(gp.quicksum(cost * abs_coeffs))

        # solve
        model.optimize()

        return coeffs.X

    @staticmethod
    def __collect_results(simplex_indicies, coeffs):
        '''
        Take the results of the optimization problem and turn it into a returnable cycle rep
        '''
        optimal_bounding_chain = simplex_indicies.assign(coefficient=coeffs) # add eoffcients
        optimal_bounding_chain = optimal_bounding_chain[optimal_bounding_chain['coefficient'] != 0].reset_index(drop=True) # keep only nonzero entries

        return optimal_bounding_chain[['simplex', 'filtration', 'coefficient']]
