import tensorflow as tf

def train_model(model, train_dataset, valid_dataset, epochs, learning_rate, weight_decay, name):
    # model loss and optimizer
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, weight_decay=weight_decay)        
    
    # model compile
    model.compile(optimizer=optimizer, loss=loss, metrics=["sparse_categorical_accuracy"])
    
    # model train
    model.fit(train_dataset, epochs=epochs, validation_data=valid_dataset)    
    
    # save model weights
    model_name = f"{name}.h5"
    model.save_weights(model_name)
    print(f"Model weights saved to {model_name}")
