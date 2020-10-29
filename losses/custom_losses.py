import tensorflow as tf
#import keras
#import tensorflow.keras.backend as K
#import tensorflow.keras.losses.BinaryCrossentropy

def custom_categorical_crossentropy():
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true, y_pred):

        #axis = 3 is softmax
#        loss_val = K.categorical_crossentropy(y_true, y_pred, from_logits=False, axis=3)
#        bce = tf.keras.losses.BinaryCrossentropy()

        result = tf.keras.backend.binary_crossentropy(y_true, y_pred, from_logits=False)

        mean_loss = tf.math.reduce_mean(result)
               
#        print("shape", mean_loss)

        return mean_loss

    # Return a function
    return loss
        
        
        