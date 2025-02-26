# LUD-Project
---
## Project1: Image Classification on CIFAR-10
1. Build dataset
    1. Unzip cifar-10 file
    2. Learn unpickle function
    3. Preprocess data
    4. Split data_batch into train_data(80%) and validation_data(20%)
2. Build Neural Network(CNN)
    1. Define __init__ function
    2. Complete forward function based on its structure(Conv1,Pool...)
3. Train my network
    1. Using Adam, lr = 0.1. **Training result**: Epoch [10/10], Validation Loss: 2.3141, Validation Accuracy: 9.44%. **Testing result**: Test Loss: 2.3057, Test Accuracy: 10.00%
    2. Using Adam, lr = 0.01. **Train result**: Epoch [10/10], Validation Loss: 1.4825, Validation Accuracy: 46.40%. **Testing result**: Test Loss: 1.4588, Test Accuracy: 47.17%
    3. Using Adam, lr = 0.001. **Training result**: Epoch [10/10], Validation Loss: 1.0713, Validation Accuracy: 71.47%. **Testing result**: Test Loss: 1.0719, Test Accuracy: 71.47%
    4. Using SGD, lr = 0.1. **Training result**: Epoch [10/10], Validation Loss: 1.0357, Validation Accuracy: 69.71%. **Testing result**: Test Loss: 1.0472, Test Accuracy: 69.36%
    5. Using SGD, lr = 0.01. **Training result**: Epoch [10/10], Validation Loss: 1.1916, Validation Accuracy: 57.63%. **Testing result**: Test Loss: 1.1860, Test Accuracy: 58.05%
    6. Using SGD, lr = 0.001. **Training result**: Epoch [10/10], Validation Loss: 2.0690, Validation Accuracy: 26.85%. **Testing result**: Test Loss: 2.0540, Test Accuracy: 27.85%
    7. Using Nadam, lr = 0.001. **Training result**: Epoch [10/10], Validation Loss: 0.9822, Validation Accuracy: 74.37%. **Testing result**: Test Loss: 1.0437, Test Accuracy: 72.82%
    8. Using Nadam, lr = 0.01. **Training result**: Epoch [10/10], Validation Loss: 1.4060, Validation Accuracy: 50.24%. **Testing result**: Test Loss: 1.4157, Test Accuracy: 49.89%
    9. Using Nadam, lr = 0.1. **Training result**: Epoch [10/10], Validation Loss: 2.3092, Validation Accuracy: 9.78%. **Testing result**: Test Loss: 2.3085, Test Accuracy: 10.00%