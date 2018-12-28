 # Importamos la librerias necesarias
import random

# Declaramos la clase
class Perceptron:
    def __iniit__(self, sample, exit, learn_rate=0.01, epoch_number=1000, bias=-1):
        # Atributos de la clase
        self.sample = sample # Datos de entrenamiento
        self.exit = exit # Salida esperada para cada dato
        self.learn_rate = learn_rate # Que tanto aprendera la red
        self.epoch_number = epoch_number
        self.bias = bias # Bias de la red
        self.number_sample = len(sample) # Numero de ejemplos
        self.col_sample = len(sample[0]) # Columnas de los datos
        self.weight = [] # Lista de pesos
        
    def trannig(self): # Metodo de entrenamiento
            for sample in self.sample: # Se recorren los datos de entrenamiento
                sample.insert(0, self.bias) # Se inserta el bias en la primera pocisión

            for i in range(self.col_sample):
                self.weight.append(random.random()) # Asignamos pesos aleatorios

            self.weight.insert(0, self.bias) # Insertamos el bias en los pesos

            epoch_count = 0

            while True:
                erro = False
                for i in range(self.number_sample):
                    u = 0
                    for j in range(self.col_sample + 1):
                        # Función de activación
                        u = u + self.weight[j] * self.sample[i][j] 

                    y = self.sign(u) # Comprobar el valor del umbral

                    if y != self.exit[i]:

                        for j in range(self.col_sample+1):
                            # Función de entrenamiento
                            # w = w + N(d(k)-y) x(k)
                            self.weight[j] = self.weight[j] + self.learn_rate * (self.exit[i]-y) * self.sample[i][j]
                        erro = True

                epoch_count = epoch_count+1 # Se aumenta el numero de epoch

                if erro == False:
                    print(('\nEpoch: \n', epoch_count)) # Mostramos el valor de epoch
                    print('-'*20)
                    print("\n")
                    break

    def sort(self, sample):
            """
            Se inserta el bias, ya que como discutimos antes,
            sera una neurona que siempre estara activada.
            """
            sample.insert(0, self.bias)
            u = 0
            for i in range(self.col_sample + 1):
                # Función de activación
                u = u + self.weight[i] * sample[i]

            # Comprobamos el valor de la función de activación
            y = self.sign(u) 

            # Si y es igual a -1, la clasificación corresponde a P1
            if  y == -1:
                print(('Ejemplo: ', sample))
                print('Clasificación: P1')
            # Si y es igual a 1, la clasificación corresponde a P1
            elif y == 1:
                print(('Ejemplo: ', sample))
                print('Clasificación: P2')

    def sign(self, u):
        return 1 if u >= 0 else -1
    
# Datos de entrenamiento
samples = [
    [0, 2],
    [-2, 2],
    [0, -2],
    [2, 0],
    [-2,2],
    [-2,-2],
    [2,-2],
    [2,2],
]

# Clasificación de los datos de entrenamiento (salidas que esperamos para cada conjunto de dato)
"""
[0,2] = 1
[-2,-2] = 1
[0,-2] = 0
...
"""
exit = [1, 1, 0, 0, 1, 1, 0, 1]

#Intancia de nuestra neurona
network = Perceptron(sample=samples, exit = exit, learn_rate=0.01, epoch_number=1000, bias=-1)
 
# Entrenamos a la neurona
network.trannig()

"""
Le pedimos al usuario datos para entrenar.
Luego mostramos el resultados
"""
while True:
    sample = []
    for i in range(2):
        sample.insert(i, float(input('Valor: ')))
    network.sort(sample) # Clasificacipon de nuevos datos
    print("\n")
