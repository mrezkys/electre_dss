import numpy as np

class ElectreSolver:
    """
    Implements the ELECTRE I method for multi-criteria decision making.

    Attributes:
        decision_matrix (np.ndarray): Matrix of alternatives (rows) vs criteria (columns).
        weights (np.ndarray): Weights for each criterion.
        criteria_types (list): List of strings ('cost' or 'benefit') for each criterion.
        alternative_names (list, optional): Names of the alternatives.
        criteria_names (list, optional): Names of the criteria.
    """
    def __init__(self, decision_matrix: np.ndarray, weights: np.ndarray, criteria_types: list, alternative_names: list = None, criteria_names: list = None):
        self.decision_matrix = np.array(decision_matrix, dtype=float)
        self.weights = np.array(weights, dtype=float)
        self.criteria_types = criteria_types

        self.num_alternatives, self.num_criteria = self.decision_matrix.shape

        if len(self.weights) != self.num_criteria:
            raise ValueError("Number of weights must match number of criteria")
        if len(self.criteria_types) != self.num_criteria:
            raise ValueError("Number of criteria types must match number of criteria")
        if not all(ct in ['cost', 'benefit'] for ct in self.criteria_types):
            raise ValueError("Criteria types must be either 'cost' or 'benefit'")

        self.alternative_names = alternative_names or [f"Alt{i+1}" for i in range(self.num_alternatives)]
        self.criteria_names = criteria_names or [f"Crit{j+1}" for j in range(self.num_criteria)]

        if len(self.alternative_names) != self.num_alternatives:
             raise ValueError("Number of alternative names must match number of alternatives")
        if len(self.criteria_names) != self.num_criteria:
             raise ValueError("Number of criteria names must match number of criteria")

        # Intermediate results storage
        self._raw_normalized_matrix = None # ADDED: Store raw r_ij
        self._weighted_matrix = None
        self._concordance_matrix = None
        self._discordance_matrix = None
        self._concordance_threshold = None
        self._discordance_threshold = None
        self._concordance_dominance_matrix = None
        self._discordance_dominance_matrix = None
        self._aggregate_dominance_matrix = None
        self._ranking = None

        # ADDED: Detailed storage for report-like output
        self._concordance_sets = {}
        self._discordance_sets = {}
        self._discordance_max_diffs = {}
        self._discordance_global_max_diff = None

    def _normalize(self):
        """Performs Euclidean normalization on the decision matrix."""
        norms = np.sqrt(np.sum(self.decision_matrix**2, axis=0))
        # Avoid division by zero if a criterion column is all zeros
        norms[norms == 0] = 1e-9 # Replace 0 with a very small number
        self._raw_normalized_matrix = self.decision_matrix / norms # Store raw r_ij

    def _calculate_weighted_matrix(self):
        """Calculates the weighted normalized decision matrix."""
        if self._raw_normalized_matrix is None:
            self._normalize()
        # Use the stored raw normalized matrix
        self._weighted_matrix = self._raw_normalized_matrix * self.weights

    def _calculate_concordance(self):
        """Calculates the concordance matrix and stores concordance sets."""
        if self._weighted_matrix is None:
            self._calculate_weighted_matrix()

        self._concordance_matrix = np.zeros((self.num_alternatives, self.num_alternatives))
        self._concordance_sets = {} # Initialize/clear

        for k in range(self.num_alternatives):
            for l in range(self.num_alternatives):
                if k == l: continue
                concordance_sum = 0
                current_concordance_set = [] # Store criteria names for this pair
                for j in range(self.num_criteria):
                    vkj = self._weighted_matrix[k, j]
                    vlj = self._weighted_matrix[l, j]
                    is_concordant = False
                    if self.criteria_types[j] == 'benefit':
                        if vkj >= vlj:
                            is_concordant = True
                    elif self.criteria_types[j] == 'cost':
                        if vkj <= vlj:
                            is_concordant = True
                    
                    if is_concordant:
                            concordance_sum += self.weights[j]
                            current_concordance_set.append(self.criteria_names[j]) # Store name

                self._concordance_matrix[k, l] = concordance_sum
                self._concordance_sets[(k, l)] = current_concordance_set # Store the set

    def _calculate_discordance(self):
        """Calculates the discordance matrix and stores related details."""
        if self._weighted_matrix is None:
            self._calculate_weighted_matrix()

        self._discordance_matrix = np.zeros((self.num_alternatives, self.num_alternatives))
        self._discordance_sets = {}
        self._discordance_max_diffs = {}
        max_diff_overall = 0

        # First pass: find the overall maximum absolute difference across all pairs and criteria
        for k in range(self.num_alternatives):
            for l in range(self.num_alternatives):
                if k == l: continue
                diffs = np.abs(self._weighted_matrix[k, :] - self._weighted_matrix[l, :])
                # Need to handle empty diffs if only one criterion?
                if diffs.size > 0:
                    current_max_diff = np.max(diffs)
                if current_max_diff > max_diff_overall:
                    max_diff_overall = current_max_diff
                # else: handle case with 0 or 1 criterion? max_diff_overall remains 0

        self._discordance_global_max_diff = max_diff_overall if max_diff_overall > 0 else 1.0

        # Second pass: calculate discordance index and store sets/diffs
        for k in range(self.num_alternatives):
            for l in range(self.num_alternatives):
                if k == l: continue
                discordance_set_diffs = []
                current_discordance_set = []
                for j in range(self.num_criteria):
                    vkj = self._weighted_matrix[k, j]
                    vlj = self._weighted_matrix[l, j]
                    is_discordant = False
                    if self.criteria_types[j] == 'benefit':
                        if vkj < vlj:
                            is_discordant = True
                    elif self.criteria_types[j] == 'cost':
                        if vkj > vlj:
                            is_discordant = True
                    
                    if is_discordant:
                        current_discordance_set.append(self.criteria_names[j])
                        discordance_set_diffs.append(np.abs(vkj - vlj))

                self._discordance_sets[(k, l)] = current_discordance_set
                if not discordance_set_diffs: # If discordance set is empty
                    self._discordance_matrix[k, l] = 0.0
                    self._discordance_max_diffs[(k, l)] = 0.0
                else:
                    max_discordance_diff = np.max(discordance_set_diffs)
                    self._discordance_max_diffs[(k, l)] = max_discordance_diff
                    # Use the stored global max diff
                    self._discordance_matrix[k, l] = max_discordance_diff / self._discordance_global_max_diff

    def _calculate_thresholds(self):
        """Calculates the concordance and discordance thresholds."""
        if self._concordance_matrix is None or self._discordance_matrix is None:
            raise RuntimeError("Concordance and Discordance matrices must be calculated first.")

        num_pairs = self.num_alternatives * (self.num_alternatives - 1)
        if num_pairs == 0: # Handle case with only one alternative
             self._concordance_threshold = 0
             self._discordance_threshold = 1
             return

        # Sum only off-diagonal elements
        total_concordance = np.sum(self._concordance_matrix)
        total_discordance = np.sum(self._discordance_matrix)

        self._concordance_threshold = total_concordance / num_pairs
        self._discordance_threshold = total_discordance / num_pairs

    def _calculate_dominance_matrices(self):
        """Calculates the concordance and discordance dominance matrices."""
        if self._concordance_threshold is None or self._discordance_threshold is None:
             self._calculate_thresholds()

        self._concordance_dominance_matrix = (self._concordance_matrix >= self._concordance_threshold).astype(int)
        self._discordance_dominance_matrix = (self._discordance_matrix <= self._discordance_threshold).astype(int)
        # Set diagonal to 0 (or keep NaN/leave undefined as it's not used for ranking)
        np.fill_diagonal(self._concordance_dominance_matrix, 0)
        np.fill_diagonal(self._discordance_dominance_matrix, 0)

    def _calculate_aggregate_dominance(self):
        """Calculates the aggregate dominance matrix E = F * G."""
        if self._concordance_dominance_matrix is None or self._discordance_dominance_matrix is None:
            self._calculate_dominance_matrices()

        self._aggregate_dominance_matrix = self._concordance_dominance_matrix * self._discordance_dominance_matrix

    def solve(self):
        """Runs the full ELECTRE I analysis."""
        self._normalize()
        self._calculate_weighted_matrix()
        self._calculate_concordance()
        self._calculate_discordance()
        self._calculate_thresholds()
        self._calculate_dominance_matrices()
        self._calculate_aggregate_dominance()
        self._determine_ranking()

    def _determine_ranking(self):
        """Determines the final ranking based on the aggregate dominance matrix."""
        if self._aggregate_dominance_matrix is None:
            self._calculate_aggregate_dominance()

        # Simple ranking: count how many others each alternative dominates
        # Alternatives that dominate more are ranked higher.
        # A more robust approach might be needed for complex scenarios (cycles)
        dominance_scores = np.sum(self._aggregate_dominance_matrix, axis=1)
        # Count how many dominate *this* alternative
        dominated_scores = np.sum(self._aggregate_dominance_matrix, axis=0)
        
        # Combine scores: prioritize dominating others, penalize being dominated
        # This is a simple heuristic; more sophisticated graph analysis could be used.
        net_scores = dominance_scores - dominated_scores

        # Rank based on net score (higher is better)
        ranked_indices = np.argsort(net_scores)[::-1]
        self._ranking = [self.alternative_names[i] for i in ranked_indices]

    def get_ranking(self):
        """Returns the final ranking of alternatives."""
        if self._ranking is None:
            self.solve() # Ensure calculation is done
        return self._ranking

    def get_intermediate_results(self):
        """Returns a dictionary containing intermediate and detailed calculation results."""
        return {
            "raw_normalized_matrix": self._raw_normalized_matrix,
            "weighted_matrix": self._weighted_matrix,
            "concordance_matrix": self._concordance_matrix,
            "discordance_matrix": self._discordance_matrix,
            "concordance_threshold": self._concordance_threshold,
            "discordance_threshold": self._discordance_threshold,
            "concordance_dominance_matrix": self._concordance_dominance_matrix,
            "discordance_dominance_matrix": self._discordance_dominance_matrix,
            "aggregate_dominance_matrix": self._aggregate_dominance_matrix,
            # Detailed additions
            "concordance_sets": self._concordance_sets,
            "discordance_sets": self._discordance_sets,
            "discordance_max_diffs": self._discordance_max_diffs,
            "discordance_global_max_diff": self._discordance_global_max_diff,
        } 