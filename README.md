# Explainable Deep Learning - Gender Classification with GradCAM 

Face recognition technology has become deeply embedded in modern life — from Apple Face ID and identity verification, to cashless payments and airport security. 

In this project, I used a pretrained ResNet-50 model to classify gender from facial images. The image samples were taken from the [FairFace dataset on Kaggle](https://www.kaggle.com/datasets/abdulwasay551/fairface-race), which includes diverse faces across race, age, and gender.

To interpret the model’s inner workings, I applied Grad-CAM and two of its enhanced variants, Grad-CAM++ and XGrad-CAM, to visualize the class-discriminative regions within each image. These techniques highlight which parts of a face contribute most to the model’s final classification decision, helping analyze how trustworthy the predictions are. 

## 📚 Methods

* **Model:** ResNet-50 pretrained on ImageNet
* **Dataset:** [FairFace (Kaggle)](https://www.kaggle.com/datasets/abdulwasay551/fairface-race)
* **Explainability Tools:**
  * Grad-CAM
  * Grad-CAM++
  * XGrad-CAM
* **Framework:** PyTorch with `torchvision` and `pytorch-grad-cam`

## ⚙️ Key Implementations

1. Model Setup
2. Image Preprocessing
3. Grad-CAM Visualization

## 🔍 Key Findings

- 5 out of 8 predictions (62.5%) were correct, among which 3 out of 4 women misclassified as men.
- All predictions hover around 50-55%, indicating high uncertainty.
- When making decisions about gender classification, model considered factors such as clothing color, sports contexts, cultural headwear, and background elements rather than facial features alone.

## 🔬 Grad-CAM Variants Comparison
- GradCAM tends to produce broader, more diffuse activation maps that spread across larger regions of the face, including background elements. This can make it difficult to pinpoint specific features driving the decision.
- GradCAM++ generates more focused attention maps with sharper boundaries, concentrating on specific facial features such as the central face region, eyes, and lower face. The heat maps show clearer differentiation between highly relevant and less relevant regions.
- XGradCAM provides the most balanced visualization, often highlighting multiple distinct facial regions (face, hair, shoulders) with relatively equal emphasis. This approach captures a more holistic view of the features contributing to the prediction.
