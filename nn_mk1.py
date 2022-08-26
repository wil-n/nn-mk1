from numpy import exp, array, random, dot






class neuralNetwork():
    def __init__(self):
        self.synaptic_weights1 = []
        self.synaptic_weights2 = []
        self.synaptic_weights3 = []
        inputs = 3
        l2 = 5
        l3 = 4

        #- Randomized weights of the network
        
        self.synaptic_weights1 = 2*random.random((inputs, l2))-1
            
        self.synaptic_weights2 = 2*random.random((l2, l3))-1
    
        self.synaptic_weights3 = 2*random.random((l3, 1))-1


    #- Sigmoid function. 1/(1 + e**-x)
    #- e = eulers number
    def __sigmoid(self, x):
        return 1/(1+exp(-x))
    def __sigmoid_derivative(self, x):
        return x*(1-x)
    

    def train(self, training_set_inputs, training_set_outputs, training_iterations):
        for iteration in range(training_iterations):

            #- Training of the network
            a2 = self.__sigmoid(dot(training_set_inputs, self.synaptic_weights1))
            a3 = self.__sigmoid(dot(a2, self.synaptic_weights2))
            output = self.__sigmoid(dot(a3, self.synaptic_weights3))

            #- error
            del4 = (training_set_outputs - output)*self.__sigmoid_derivative(output)

            #errors in each layer
            del3 = dot(self.synaptic_weights3, del4.T)*(self.__sigmoid_derivative(a3).T)
            del2 = dot(self.synaptic_weights2, del3)*(self.__sigmoid_derivative(a2).T)

            #- get adjustments for each layer
            adjustment3 = dot(a3.T, del4)
            adjustment2 = dot(a2.T, del3.T)
            adjustment1 = dot(training_set_inputs.T, del2.T)


            #- adjust weights
            self.synaptic_weights1 = adjustment1
            self.synaptic_weights2 = adjustment2
            self.synaptic_weights3 = adjustment3
            
    def forward_pass(self, inputs):
        #- pass
        a2 = self.__sigmoid(dot(inputs, self.synaptic_weights1))
        a3 = self.__sigmoid(dot(a2, self.synaptic_weights2))
        output = self.__sigmoid(dot(a3, self.synaptic_weights3)) 
        return output

if __name__ == "__main__":
    neural_network = neuralNetwork()

    print("Random starting synaptic weights (layer 1): ")
    print(neural_network.synaptic_weights1)
    print("\nRandom starting synaptic weights (layer 2): ")
    print(neural_network.synaptic_weights2)
    print("\nRandom starting synaptic weights (layer 3): ")
    print(neural_network.synaptic_weights3)

    #training_set_inputs = array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1]])
    #training_set_outputs = array([[0, 0, 1, 1, 1, 1, 1, 1, 1]]).T

    training_set_inputs = array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1]])
    training_set_outputs = array([[1, 1, 1, 1, 1, 1]]).T
    neural_network.train(training_set_inputs, training_set_outputs, 100)


    print("\nNew synaptic weights (layer 1) after training: ")
    print(neural_network.synaptic_weights1)
    print("\nNew synaptic weights (layer 2) after training: ")
    print(neural_network.synaptic_weights2)
    print("\nNew synaptic weights (layer 3) after training: ")
    print(neural_network.synaptic_weights3)



    print("\n\nOutput: ")
    print(neural_network.forward_pass(array([1, 0, 1])))
   
    



