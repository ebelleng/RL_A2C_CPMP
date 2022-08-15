from tensorflow import keras
from permutational_layer import *
from tensorflow.keras.optimizers import Adam

from data import *


def create_critic_model(rows=7, columns=7):
    perm_layer1 = PermutationalLayer(PermutationalEncoder(PairwiseModel((rows+2,), repeat_layers(Dense, [32, 32], activation='relu')), columns), name='permutational_layer1')
    perm_layer2 = PermutationalLayer(PermutationalEncoder(PairwiseModel((32,), repeat_layers(Dense, [32, 32], activation='relu')), columns), name='permutational_layer2')
    perm_layer3 = PermutationalLayer(PermutationalEncoder(PairwiseModel((32,), repeat_layers(Dense, [32, 32], activation='relu')), columns), pooling=maximum, name='permutational_layer3')

    dense1 = Dense(128, activation="relu", name="layer1")
    dense2 = Dense(64, activation="relu", name="layer2")
    dense3 = Dense(1, activation="linear", name="layer3")

    inputs = []
    for i in range(columns):
        inputs.append(Input((rows+2,), name='stack_'+str(i)))

    outputs = perm_layer1(inputs)
    outputs = perm_layer2(outputs)
    outputs = perm_layer3(outputs)
    outputs = dense1(outputs)
    outputs = dense2(outputs)
    output = dense3(outputs)

    model = Model(inputs, output)

    adam  = Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=adam, metrics=['mse'])

    return model

def train_critic_model(rows, columns, critic_file):
    critic_model = create_critic_model(rows, columns) 

    for i in range(200):
        X,y = generate_data_greedy(51200)
        Xtrain_input_array = InputEncoder(X, rows, columns)
        history = critic_model.fit(Xtrain_input_array, y, epochs=10, batch_size=1024, verbose=False)
        print(f'Iter {i:-2}: ', end=' ')
        print(history.history['loss'][0])
        critic_model.save(critic_file)

    return critic_model

def test_critic_model(critic_model, rows=7, columns=7):
    X,y = generate_data_greedy(200)
    X_train = InputEncoder(X, rows, columns)
    critic_model.evaluate(X_train, y)

def load_critic_model(critic_file):
    return keras.models.load_model(critic_file)

def main(model_type=''):
    rows, columns = (7,7)
    critic_file = f'critic_model_{rows}x{columns}_v2.h5'

    if model_type == 'train':
        model = train_critic_model(rows, columns, critic_file)
    else:  
        model = load_critic_model(critic_file)

    test_critic_model(model)

if __name__ == "__main__":
    #main('train')
    main()


