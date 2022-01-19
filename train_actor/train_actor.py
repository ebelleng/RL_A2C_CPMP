from keras.models import Sequential
from keras.layers import Dense, Input, Flatten
from keras.models import Model
import tensorflow as tf

from data import *

def create_actor_model(rows=7, columns=7):
    n_actions = (rows-1) * columns

    model = Sequential()

    model.add(Dense(64, activation='relu'))
    model.add(Dense(n_actions, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model

def train_actor_model(rows, columns, actor_file):
    actor_model = create_actor_model(rows, columns)
    
    X,y = generate_data_greedy(1)
    print(y)
    Xtrain_input_array = InputEncoder(X, rows, columns)
    #print(np.array(Xtrain_input_array).shape)

    X_train_tensor = tf.convert_to_tensor(Xtrain_input_array)
    print(X_train_tensor)

    actor_model.fit(X_train_tensor, y, epochs = 5)
    #actor_model.fit(X, y, epochs = 5)
    
    return actor_model

def test_actor_model(actor_model, rows=7, columns=7):
    X,y = generate_data_greedy(200)
    X_train = InputEncoder(X, rows, columns)

    X_train_tensor = tf.convert_to_tensor(X_train)

    actor_model.evaluate(X_train_tensor, y)
    
def load_actor_model(actor_file):
    return

def main(model_type='load'):
    rows, columns = (7, 7)
    actor_file = f'actor_model_{rows}x{columns}_v1.h5'

    if model_type == 'train':
        model = train_actor_model(rows, columns, actor_file)
    else:
        model = load_actor_model(actor_file)

    test_actor_model(model)


if __name__ == "__main__":
    main('train')
    #main()
