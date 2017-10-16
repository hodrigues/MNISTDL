
# Conjunto de dados MNIST
from keras.datasets import mnist
 
# Modelo sequencial keras
from keras.models import Sequential

# Camadas densas - neuronio conectado a cada neuronio da proxima camada
from keras.layers import Dense

# Modulo do Keras responsavel por varias rotinas de pre-processamento 
from keras.utils import np_utils

# Dividindo os dados em treino e teste
(X_train, y_train), (X_test, y_test) = mnist.load_data()
 
num_pixels = X_train.shape[1] * X_train.shape[2]

# Nivel de precisao 32 bits
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# Escala cinza 0-255  
X_train = X_train / 255
X_test = X_test / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Numero de tipos de digitos encontrados no MNIST.
num_classes = y_test.shape[1]
 
# Modelo de três camadas
second_layer = int(num_pixels/2) # pixels da segunda camada
third_layer = int(num_pixels/4) # pixels da terceira camada
fourth_layer = int(num_pixels/8) #pixels da quarta camada


def base_model():
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='sigmoid'))
    model.add(Dense(second_layer, input_dim=num_pixels, kernel_initializer='normal', activation='sigmoid'))
    model.add(Dense(third_layer, input_dim=second_layer, kernel_initializer='normal', activation='sigmoid'))
    model.add(Dense(fourth_layer, input_dim=third_layer, kernel_initializer='normal', activation ='sigmoid'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax', name='preds'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

model = base_model()

model.summary()

# Treinamento do modelo
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=100, verbose=2)

# Avaliando a performance
scores = model.evaluate(X_test, y_test, verbose=0)
print("Erro de: %.2f%%" % (100-scores[1]*100))
