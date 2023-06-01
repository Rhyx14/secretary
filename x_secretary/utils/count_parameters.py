def count_parameters(model)-> int:
    '''
    Count parameters in a model, if require gradient
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)