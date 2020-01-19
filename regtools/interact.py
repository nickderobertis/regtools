
from .tools import _to_list_if_tuple

def create_interaction_variables(df, interaction_tuples):
    """
    Note: inplace
    """
    interaction_tuples = _to_list_if_tuple(interaction_tuples)

    interacted_variables = [_create_interaction_variable(df, tup) for tup in interaction_tuples]

    return interacted_variables

def delete_interaction_variables(df, interaction_tuples):
    """
    Note: inplace
    """
    for_delete = [_interaction_tuple_to_var_name(tup) for tup in interaction_tuples]
    df.drop(for_delete, axis=1, inplace=True)

def _create_interaction_variable(df, interaction_tuple):
    """
    Note: inplace
    """

    full_name = _interaction_tuple_to_var_name(interaction_tuple)

    # Create base for multiplication by each variable
    df[full_name] = 1

    for col in interaction_tuple:
        df[full_name] = df[full_name] * df[col]

    return full_name

def _interaction_tuple_to_var_name(interaction_tuple):
    return ' x '.join(interaction_tuple)

def _collect_variables_from_interaction_tuples(interaction_tuples):
    items = set()
    for tup in interaction_tuples:
        [items.add(item) for item in tup]
    return list(items)