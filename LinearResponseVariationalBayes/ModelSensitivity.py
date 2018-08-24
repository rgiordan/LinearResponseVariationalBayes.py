import autograd
import autograd.numpy as np

import LinearResponseVariationalBayes as vb
import LinearResponseVariationalBayes.SparseObjectives as obj_lib

from copy import deepcopy

import math


# Append a jacobian vector product to a function.
# Args:
#  - fun: The function to be differentiated.
#  - num_base_args: The number of inputs to the base function, i.e.,
#    to the function before any differentiation.
#  - argnum: Which argument should be differentiated with respect to.
#    Must be between 0 and num_base_args - 1.
# Returns:
#   Denote the base args x1, ..., xB, where B == num_base_args.
#   Let argnum = k.  Then append_jvp returns a function,
#   fun_jvp(x1, ..., xB, ..., v) =
#     \sum_i (dfun_dx_{ki}) v_i | (x1, ..., xB).
#   That is, it returns the Jacobian vector product where the Jacobian
#   is taken with respect to xk, and the vector product is with the
#   final argument.
#
#   This function is designed to be used recursively to calculate
#   higher-order Jacobian-vector products.
def append_jvp(fun, num_base_args=1, argnum=0):
    assert argnum < num_base_args

    fun_jvp = autograd.make_jvp(fun, argnum=argnum)
    def obj_jvp_wrapper(*argv):
        # These are the base arguments -- the points at which the
        # Jacobians are evaluated.
        base_args = argv[0:num_base_args]
        # The rest of the arguments are the vectors, with which inner
        # products are taken in the order they were passed to
        # append_jvp.
        vec_args = argv[num_base_args:]
        #print(vec_args)
        if (len(vec_args) > 1):
            # Then this is being applied to an existing Jacobian
            # vector product.  The last will be the new vector, and
            # we need to evaluate the function at the previous vectors.
            # The new jvp will be appended to the end of the existing
            # list.
            old_vec_args = vec_args[:-1]
            return fun_jvp(*base_args, *old_vec_args)(vec_args[-1])[1]
        else:
            return fun_jvp(*base_args)(*vec_args)[1]

    return obj_jvp_wrapper


# A single term in a Taylor expansion of a two-parameter objective with
# methods for computing its derivatives.  The nomenclature assumes that
# we are calculating derivatives of g(eta, eps) at (eta0, eps0).  This
# can be used to calculate d^k\hat\eta / d\eps^k | (eta0, eps0) where
# \hat\eta: g(\hat\eta, \eps) = 0.
#
# Attributes:
# - eps_order:  The number of epsilon derivatives of g..
# - eta_orders: A vector of length order - 1.  Entry i contains the number
#   of terms d\eta^{i + 1} / d\epsilon^{i + 1}.
# - prefactor: The constant multiple in front of this term.
# - eval_eta_derivs: A vector of functions to evaluate d\eta^i / d\epsilon^i.
#   The functions should take arguments (eta0, eps0, deps) and the i^{th} entry should
#   evaluate d\eta^i / d\epsilon^i (deps^i) |_{eta0, eps0}.
# - eval_g_derivs: A list of lists of g jacobian vector product functions.
#   The array should be such that
#   eval_g_derivs[i][j](eta0, eps0, v1 ... vi, w1 ... wj)
#   evaluates d^{i + j} G / (deta^i)(deps^j)(v1 ... vi)(w1 ... wj).
class DerivativeTerm:
    def __init__(self, eps_order, eta_orders, prefactor,
                 eval_eta_derivs, eval_g_derivs):

        # Base properties.
        self.eps_order = eps_order
        self.eta_orders = eta_orders
        self.prefactor = prefactor
        self.eval_eta_derivs = eval_eta_derivs
        self.eval_g_derivs = eval_g_derivs

        # Derived quantities.

        # The order is the total number of epsilon derivatives.
        self.order = int(
            self.eps_order + \
            np.sum(self.eta_orders * np.arange(1, len(self.eta_orders) + 1)))

        # The derivative of g needed for this particular term.
        self.eval_g_deriv = \
            eval_g_derivs[np.sum(eta_orders)][self.eps_order]

        # Sanity checks.
        # The rules of differentiation require that these assertions be true
        # -- that is, if terms are generated using the differentiate()
        # method from other well-defined terms, these assertions should always
        # be sastisfied.
        assert isinstance(self.eps_order, int)
        assert len(self.eta_orders) == self.order
        assert self.eps_order >= 0 # Redundant
        for eta_order in self.eta_orders:
            assert eta_order >= 0
            assert isinstance(eta_order, int)
        assert len(self.eval_eta_derivs) >= self.order - 1
        assert len(eval_g_derivs) > len(self.eta_orders)
        for eta_deriv_list in eval_g_derivs:
            assert len(eta_deriv_list) > self.eps_order

    def __str__(self):
        return 'Order: {}\t{} * eta{} * eps[{}]'.format(
            self.order, self.prefactor, self.eta_orders, self.eps_order)

    def evaluate(self, eta0, eps0, deps):
        # First eta arguments, then epsilons.
        vec_args = []

        for i in range(len(self.eta_orders)):
            eta_order = self.eta_orders[i]
            if eta_order > 0:
                vec = self.eval_eta_derivs[i](eta0, eps0, deps)
                for j in range(eta_order):
                    vec_args.append(vec)

        for i in range(self.eps_order):
            vec_args.append(deps)

        return self.prefactor * self.eval_g_deriv(eta0, eps0, *vec_args)

    def differentiate(self, eval_next_eta_deriv):
        derivative_terms = []
        new_eval_eta_derivs = deepcopy(self.eval_eta_derivs)
        new_eval_eta_derivs.append(eval_next_eta_deriv)

        old_eta_orders = deepcopy(self.eta_orders)
        old_eta_orders.append(0)

        # dG / deps.
        #print('dg/deps')
        derivative_terms.append(
            DerivativeTerm(
                eps_order=self.eps_order + 1,
                eta_orders=deepcopy(old_eta_orders),
                prefactor=self.prefactor,
                eval_eta_derivs=new_eval_eta_derivs,
                eval_g_derivs=self.eval_g_derivs))

        # dG / deta.
        #print('dg/deta')
        new_eta_orders = deepcopy(old_eta_orders)
        new_eta_orders[0] = new_eta_orders[0] + 1
        derivative_terms.append(
            DerivativeTerm(
                eps_order=self.eps_order,
                eta_orders=new_eta_orders,
                prefactor=self.prefactor,
                eval_eta_derivs=new_eval_eta_derivs,
                eval_g_derivs=self.eval_g_derivs))

        # Derivatives of each d^{i}eta / deps^i term.
        for i in range(len(self.eta_orders)):
            #print('i: ', i)
            eta_order = self.eta_orders[i]
            if eta_order > 0:
                new_eta_orders = deepcopy(old_eta_orders)
                new_eta_orders[i] = new_eta_orders[i] - 1
                new_eta_orders[i + 1] = new_eta_orders[i + 1] + 1
                derivative_terms.append(
                    DerivativeTerm(
                        eps_order=self.eps_order,
                        eta_orders=new_eta_orders,
                        prefactor=self.prefactor * eta_order,
                        eval_eta_derivs=new_eval_eta_derivs,
                        eval_g_derivs=self.eval_g_derivs))

        return derivative_terms

    # Return whether another term matches this one in the pattern of derivatives.
    def check_similarity(self, term):
        return \
            (self.eps_order == term.eps_order) & \
            (self.eta_orders == term.eta_orders)

    # Assert that another term has the same pattern of derivatives and
    # return a new term that combines the two.
    def combine_with(self, term):
        assert self.check_similarity(term)
        return DerivativeTerm(
            eps_order=self.eps_order,
            eta_orders=self.eta_orders,
            prefactor=self.prefactor + term.prefactor,
            eval_eta_derivs=self.eval_eta_derivs,
            eval_g_derivs=self.eval_g_derivs)


# Generate an array of JVPs of the two arguments of the target function fun.
#
# Args:
#   - fun: The function to be differentiated.  The first two arguments
#   should be vectors for differentiation, i.e., fun should have signature
#   fun(x1, x2, ...) and return a numeric value.
#   - order: The maximum order of the derivative to be generated.
#
# Returns:
#   An array of functions where element eval_fun_derivs[i][j] is a function
#   eval_fun_derivs[i][j](x1, x2, ..., v1, ... vi, w1, ..., wj)) =
#   d^{i + j}fun / (dx1^i dx2^j) v1 ... vi w1 ... wj.
#
# TODO: do these need to be wrapped?
def generate_two_term_derivative_array(fun, order):
    eval_fun_derivs = [[ fun ]]
    for x1_ind in range(order):
        if x1_ind > 0:
            # Append one x1 derivative.
            next_deriv = append_jvp(
                eval_fun_derivs[x1_ind - 1][0], num_base_args=2, argnum=0)
            eval_fun_derivs.append([ next_deriv ]
        for x2_ind in range(order):
            # Append one x2 derivative.
            next_deriv = append_jvp(
                eval_fun_derivs[x1_ind][x2_ind], num_base_args=2, argnum=1)
            eval_fun_derivs[x1_ind].append(next_deriv)
    return eval_fun_derivs



# Combine like derivative terms.
#
# Args:
#   - dterms: An array of DerivativeTerms.
#
# Returns:
#   - A new array of derivative terms that evaluate equivalently where
#   terms with the same derivative signature have been combined.
def consolidate_terms(dterms):
    unmatched_indices = [ ind for ind in range(len(dterms)) ]
    consolidated_dterms = []
    while len(unmatched_indices) > 0:
        match_term = dterms[unmatched_indices.pop(0)]
        for ind in unmatched_indices:
            if (match_term.eta_orders == dterms[ind].eta_orders):
                match_term = match_term.combine_with(dterms[ind])
                unmatched_indices.remove(ind)
        consolidated_dterms.append(match_term)

    return consolidated_dterms

# Evaluate an array of derivative terms.
#
# Args:
#   - dterms: An array of derivative terms.
#   - eta0: The value of the first argument.
#   - eps0: The value of the second argument.
#   - deps: The change in epsilon by which to multiply the Jacobians.
#   - include_highest_eta_order: If true, include the term with
#   d^k eta / deps^k, where k == order.  The main use of these DerivativeTerms
#   at the time of writing is precisely to evaluate this term using the other
#   terms, and this can be accomplished by setting include_highest_eta_order
#   to False.
#
# Returns:
#   The sum of the evaluated DerivativeTerms.
def evaluate_terms(dterms, eta0, eps0, deps, include_highest_eta_order=True):
    vec = None
    for term in dterms:
        if include_highest_eta_order or (term.eta_orders[-1] == 0):
            if vec is None:
                vec = term.evaluate(eta0, eps0, deps)
            else:
                vec += term.evaluate(eta0, eps0, deps)
    return vec
