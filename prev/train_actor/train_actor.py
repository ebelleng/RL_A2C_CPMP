from keras.models import Sequential
from keras.layers import Dense, Input, Flatten
from keras.models import Model
import tensorflow as tf

from data import *

def create_actor_model(rows=7, columns=7):
    n_actions = (rows-1) * columns

    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation = tf.nn.relu),
                tf.keras.layers.Dense(n_actions, activation = tf.nn.softmax)])
    
    model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics =['accuracy'])

    return model

def train_actor_model(rows, columns, actor_file):
    actor_model = create_actor_model(rows, columns)
    
    X,y = generate_data_greedy(1)
    
    actor_model.fit(X, y, epochs = 5)
    
    return actor_model

def test_actor_model(actor_model, rows=7, columns=7):
    X,y = generate_data_greedy(200)
   
    actor_model.evaluate(X, y)
    
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
