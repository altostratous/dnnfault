import re

from tensorflow.python.keras import Model
from tensorflow.python.keras.engine.input_layer import InputLayer


def insert_layer_nonseq(model, layer_regex, insert_layer_factory,
                        insert_layer_name=None, position='after', model_name=None, only_last_node=False):

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        if only_last_node:
            nodes = layer._outbound_nodes[-1:]
        else:
            nodes = layer._outbound_nodes
        for node in nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    for i, layer in enumerate(model.layers):
        if isinstance(layer, InputLayer) or i == 0:
            network_dict['new_output_tensor_of'].update(
                    {layer.name: layer.input})

    # Iterate over all layers after the input
    model_outputs = []
    for layer in model.layers:
        if layer.name in network_dict['new_output_tensor_of']:
            continue
        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux]
                for layer_aux in network_dict['input_layers_of'][layer.name]]

        # Insert layer if name matches the regular expression
        if re.match(layer_regex, layer.name):
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                try:
                    x = layer(*layer_input)
                except TypeError as t:
                    if 'arguments' in str(t):
                        x = layer(layer_input)
                    else:
                        raise t
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            insert_layer_name = '{}_{}'.format(layer.name,
                                               insert_layer_name)
            new_layer = insert_layer_factory(insert_layer_name)
            x = new_layer(x)
            # print('New layer: {} Old layer: {} Type: {}'.format(new_layer.name,
            #                                                 layer.name, position))
            if position == 'before':
                x = layer(x)
        else:
            try:
                x = layer(*layer_input)
            except TypeError as t:
                if 'arguments' in str(t):
                    x = layer(layer_input)
                else:
                    raise t

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
        if layer.name in model.output_names:
            model_outputs.append(x)

    kwargs = {}
    if model_name is not None:
        kwargs['name'] = model_name
    return Model(inputs=model.inputs, outputs=model_outputs, **kwargs)
