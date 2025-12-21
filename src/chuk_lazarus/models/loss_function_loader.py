import importlib


def load_loss_function(loss_function_name):
    # get the module and function name
    module_name, function_name = loss_function_name.rsplit(".", 1)

    # import the module
    module = importlib.import_module(module_name)

    # return the function
    return getattr(module, function_name)
