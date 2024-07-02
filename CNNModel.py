import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

class CCNModel:
    def __init__(self):
        # Set random seed for reproducibility
        tf.random.set_seed(42)
        # Define constants
        self.IMG_HEIGHT, self.IMG_WIDTH = 224, 224
        self.BATCH_SIZE = 32
        self.EPOCHS = 10

        # Define data directories
        self.train_dir = 'train'
        self.validation_dir = 'valid'
        self.test_dir = 'test'

        self.prepareData()

    def __init__(self, model):
        # Set random seed for reproducibility
        tf.random.set_seed(42)
        # Define constants
        self.IMG_HEIGHT, self.IMG_WIDTH = 224, 224
        self.BATCH_SIZE = 32
        self.EPOCHS = 10

        # Define data directories
        self.train_dir = 'train'
        self.validation_dir = 'valid'
        self.test_dir = 'test'

        self.prepareData()
        self.model = load_model(model)


    def prepareData(self):
        # Data augmentation and preprocessing
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        validation_datagen = ImageDataGenerator(rescale=1./255)

        # Load and prepare the data
        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
            batch_size=self.BATCH_SIZE,
            class_mode='binary'
        )

        self.validation_generator = validation_datagen.flow_from_directory(
            self.validation_dir,
            target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
            batch_size= self.BATCH_SIZE,
            class_mode='binary'
        )
        # Prepare the test data
        test_datagen = ImageDataGenerator(rescale=1./255)
        self.test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
            batch_size=self.BATCH_SIZE,
            class_mode='binary'
        )

    def createModel(self):
        # Create the model
        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        # Compile the model
        self.model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

        trainModel()

    def trainModel(self):
        # Train the model
        history = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.train_generator.samples // self.BATCH_SIZE,
            epochs=self.EPOCHS,
            validation_data=self.validation_generator,
            validation_steps=self.validation_generator.samples // self.BATCH_SIZE
        )
        # Save the model
        self.model.save('malaria_detection_model.h5')

    def plotModelData(self):
        # # Plot training history
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend()
        plt.title('Model Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.title('Model Loss')
        plt.show()

    # Function to predict on a single image
    def predict_image(self,image_path):
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array /= 255.

        prediction = self.model.predict(img_array)
        return "Infected" if prediction[0][0] > 0.5 else "Uninfected"

    def evaluateModel(self):
        # Evaluate the model
        test_loss, test_accuracy = self.model.evaluate(self.test_generator)
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")



model = CCNModel('malaria_detection_model.h5')
model.evaluateModel()