function net = buildNet(options)
%=========================================================================%
% Build the neural network.
%=========================================================================%

switch options.type
    case 'MLP1'
        input = imageInputLayer(options.inputSize, 'Name', 'input');
        fc1 = fullyConnectedLayer(2048, 'Name', 'fc1');
        elu1 = eluLayer('Name','elu1');
        drop1 = dropoutLayer(0.4, 'Name', 'drop1');
        fc2 = fullyConnectedLayer(2048, 'Name', 'fc2');
        elu2 = eluLayer('Name','elu2');
        drop2 = dropoutLayer(0.4, 'Name', 'drop2');
        fc3 = fullyConnectedLayer(2048, 'Name', 'fc3');
        elu3 = eluLayer('Name','elu3');
        drop3 = dropoutLayer(0.4, 'Name', 'drop3');
        fc4 = fullyConnectedLayer(2048, 'Name', 'fc4');
        elu4 = eluLayer('Name','elu4');
        drop4 = dropoutLayer(0.4, 'Name', 'drop4');
        fc5 = fullyConnectedLayer(2048, 'Name', 'fc5');
        elu5 = eluLayer('Name','elu5');
        drop5 = dropoutLayer(0.4, 'Name', 'drop5');
        fc6 = fullyConnectedLayer(options.numAnt(2), 'Name', 'fc6');
        sfm = softmaxLayer('Name','sfm');
        classifier = classificationLayer('Name','classifier');

        layers = [
                  input
                  fc1
                  elu1
                  drop1
                  fc2
                  elu2
                  drop2
                  fc3
                  elu3
                  drop3
                  fc4
                  elu4
                  drop4
                  fc5
                  elu5
                  drop5
                  fc6
                  sfm
                  classifier
                 ];
        net = layerGraph(layers);

end
