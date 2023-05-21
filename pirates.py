import tensorflow as tf
import pickle as pkl

#Data Retrieval
#Originally, I had used the folder of Sonar Images and parsed through them and 
#manually flattened the picture into pixels on a gray scale, but that was taking
#a long time so I decided to use pickle to store the formatted data. This was also
#convient since I need fewer libraries for this code to work and I didn't have to 
#convert to numpy every time.


pickle_x = open('ShipWreckIns.pickle', 'rb') #Gets file of numpy matrices which are inputs
x = pkl.load(pickle_x)
pickle_x.close()

pickle_y = open('ShipWreckOuts.pickle', 'rb') #Gets file of outputs stored in an numpy array
y = pkl.load(pickle_y)
pickle_y.close()

x_train, x_test = x[:2000], x[2000:2500] #Splits data 2000 for training and 500 for testing
y_train, y_test = y[:2000], y[2000:2500] #Which is a 80% Training and 20% Testing split

x_train, x_test = x_train / 255.0, x_test / 255.0 #Normalizes the data to a scale of 0-1

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(507, 1024)),
  tf.keras.layers.Dense(256, activation='relu'),  # Hidden Layer 1 with 256 nodes
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(128, activation='relu'),  # Hidden Layer 2 with 128 nodes
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(64, activation='relu'),   # Hidden Layer 3 with 64 nodes
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(2)                        # Output Layer
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) #Initalizes the error function

model.compile(optimizer='adam', #Uses a modified backprop algorithim
              loss=loss_fn, #Uses the previously initialized error function
              metrics=['accuracy']) #Used to store accuracy data

model.fit(x_train, y_train, epochs=25) #Runs through 25 epochs

model.evaluate(x_test,  y_test, verbose=2) #Tests the running model

probability_model = tf.keras.Sequential([ #Attachs softmax function to return a probability
  model,
  tf.keras.layers.Softmax()
])

probability_model(x_test[:5])