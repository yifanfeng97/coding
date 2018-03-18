
def get_state_dict(model_state_dict, from_state_dict):
    model_param = {k: v for k, v in from_state_dict.items() if k in model_state_dict and k.find('fc') == -1}
    model_state_dict.update(model_param)
    return model_state_dict