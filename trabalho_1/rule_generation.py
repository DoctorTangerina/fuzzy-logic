import pandas as pd
import numpy as np
import skfuzzy as fuzz
import sympy as sp
import sklearn as sk
from skfuzzy import control as ctrl

def select_higher_membership(linguistic_variable, input_value):
    p = 0
    label_max = None
    input_value = np.clip(input_value, linguistic_variable.universe.min(), linguistic_variable.universe.max())
    for label, set in linguistic_variable.terms.items():
        pertinence = fuzz.interp_membership(linguistic_variable.universe, set.mf, input_value)
        if pertinence > p:
            label_max = label 
            p = pertinence 
    return label_max, p

def create_rule(list_values, list_variables, n_outputs = 1):
    rule_p = 1
    list_antecedents = []
    list_consequents = []
    n_inputs = len(list_variables) - n_outputs
    for (variable, value) in zip(list_variables[:n_inputs], list_values[:n_inputs]):
        label, p = select_higher_membership(variable, value)
        rule_p *= p
        list_antecedents.append(variable[label])
    for (variable, value) in zip(list_variables[n_inputs:], list_values[n_inputs:]):
        # compute consequents based on the variables that come after the antecedents
        label, p = select_higher_membership(variable, value)
        list_consequents.append(variable[label])
    return [list_antecedents, list_consequents, rule_p]

def create_rules(df, list_variables, n_outputs=1):
    rule_list = []
    for i in range(len(df)):
        rule_list.append(create_rule(df.iloc[i], list_variables, n_outputs))
    return rule_list

def filter_rules(list):
    list_filtered = []
    for i in range(len(list)):
        if list[i][0] not in [x[0] for x in list_filtered]:
            list_filtered.append(list[i])
        else:
            for j in range(len(list_filtered)):
                if list_filtered[j][0] == list[i][0]:
                    if list_filtered[j][2] < list[i][2]:
                        list_filtered[j] = list[i]
    return list_filtered

def create_fuzzy_rules(rules, and_func):
    list_rules = []
    for rule in rules:
        combined_antecedent = rule[0][0]
        for element in rule[0][1:]:
            combined_antecedent &= element
        combined_consequent = rule[1][0]
        for element in rule[1][1:]:
            combined_consequent &= element
        list_rules.append(ctrl.Rule(combined_antecedent, combined_consequent, and_func=and_func))
    return list_rules

def create_fuzzy_system(df, list_variables, and_func=np.fmin, n_outputs=1):
    rules = create_rules(df, list_variables, n_outputs)
    rules = filter_rules(rules)
    rules = create_fuzzy_rules(rules, and_func)
    return ctrl.ControlSystem(rules)


def _format_rule_side(side):
    """Return a readable string for a Rule antecedent or consequent side.

    The function tries several common attribute patterns from skfuzzy
    compound terms and falls back to `str()` if nothing else matches.
    """
    if side is None:
        return ""

    # Prefer the built-in string representation if it looks informative
    try:
        s = str(side)
        if s and ("if" in s.lower() or "then" in s.lower() or " is " in s.lower()):
            return s
    except Exception:
        pass

    parts = []

    # If side exposes a mapping of inputs -> terms (common in some skfuzzy internals)
    if hasattr(side, 'inputs') and isinstance(getattr(side, 'inputs'), dict):
        for var, term in side.inputs.items():
            var_name = getattr(var, 'label', None) or getattr(var, 'name', None) or str(var)
            term_label = getattr(term, 'label', None) or str(term)
            parts.append(f"{var_name} IS {term_label}")
        return ' AND '.join(parts)

    # If side exposes a sequence of terms (could be .terms, .components, .args, etc.)
    candidates = None
    for attr in ('terms', 'components', 'args'):
        if hasattr(side, attr):
            candidates = getattr(side, attr)
            break

    if candidates is None and hasattr(side, '__iter__') and not isinstance(side, (str, bytes)):
        candidates = side

    if candidates is not None:
        for t in candidates:
            # Some internals use (variable, term) tuples
            if isinstance(t, tuple) and len(t) >= 2:
                var, term = t[0], t[1]
            else:
                var = getattr(t, 'variable', None)
                term = getattr(t, 'term', None) or getattr(t, 'label', None) or t

            var_name = getattr(var, 'label', None) or getattr(var, 'name', None) or str(var)
            term_label = getattr(term, 'label', None) if hasattr(term, 'label') else str(term)
            parts.append(f"{var_name} IS {term_label}")

        if parts:
            return ' AND '.join(parts)

    # Final fallback
    try:
        return str(side)
    except Exception:
        return repr(side)


def pretty_print_rules(system_or_rules):
    """Pretty-print rules from a `ctrl.ControlSystem` or an iterable of `ctrl.Rule`.

    Usage example:
    `sys = create_fuzzy_system(df, list_variables)`
    `pretty_print_rules(sys)`

    The function attempts to produce readable "IF ... THEN ..." lines and is
    defensive about different internal representations used by skfuzzy.
    """
    # Accept either a ControlSystem (has .rules) or a list/iterable of rules
    if hasattr(system_or_rules, 'rules'):
        rules = getattr(system_or_rules, 'rules')
    else:
        rules = system_or_rules

    # If there are no rules, print a short message
    try:
        iterator = iter(rules)
    except TypeError:
        print('No rules found in the provided object.')
        return

    for i, rule in enumerate(rules, start=1):
        ant = _format_rule_side(getattr(rule, 'antecedent', None))
        cons = _format_rule_side(getattr(rule, 'consequent', None))
        # Some Rule implementations include a readable __str__ already; include it as fallback
        try:
            base = str(rule)
            if base and ("if" in base.lower() or "then" in base.lower()):
                print(f"Rule {i}: {base}")
                continue
        except Exception:
            pass

        print(f"Rule {i}: IF {ant} THEN {cons}")