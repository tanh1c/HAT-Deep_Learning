# Streamlit Demo Script for Image Track

App URL:
`https://hat-deeplearning-tjflfkryrwcwsas6py8ta5.streamlit.app/`

This script is designed for a short live demo after the image-result slides.

---

## 1. Very Short Demo Version

### What to do on screen
1. Open the Streamlit app.
2. Show the four available models.
3. Select `ViT-B/16 · Head 3 + Full 8`.
4. Upload one dog image.
5. Click `Predict & Explain`.
6. Show:
   - predicted breed
   - top-1 confidence
   - top-k predictions
   - interpretability image

### What to say
"Besides offline experiments, we also deployed a small Streamlit demo for the image track. The app includes all four benchmark checkpoints: ResNet-50 full fine-tuning, ResNet-50 staged fine-tuning, ViT-B/16 full fine-tuning, and ViT-B/16 staged fine-tuning. For this demo, I choose the best overall model, ViT-B/16 with the staged strategy. I upload a dog image, run prediction, and the app returns the predicted breed, the confidence score, the top candidate classes, and an interpretability view. For ViT, this view is now shown as attention rollout rather than a raw last-layer attention map."

---

## 2. Full Demo Script

## Demo Step A. Open the app

### What to do on screen
- Open the Streamlit URL.
- Wait until the `Stanford Dogs Image Demo` title is visible.

### What to say
"After training and evaluation, we deployed the image model as an interactive web demo using Streamlit. The purpose is to make the benchmark results more practical: instead of only reading numbers in a report, we can directly test the trained models on new dog images."

---

## Demo Step B. Introduce the four models

### What to do on screen
- Point to the `Choose model` radio options.
- Briefly hover or click through the four choices.

### What to say
"The app exposes the same four checkpoints used in our fair benchmark. We have two backbones, ResNet-50 and ViT-B/16, and each backbone has two training strategies: full fine-tuning and staged fine-tuning. This means the web demo is directly connected to the same models reported in our results tables."

### Optional short version
"Here we can test all four benchmark models, not just one final checkpoint."

---

## Demo Step C. Choose the model for the live test

### Recommended default choice
`ViT-B/16 · Head 3 + Full 8`

### Why this choice
- Best accuracy in the benchmark
- Best macro F1
- Strong calibration

### What to say
"For the live demo, I will use ViT-B/16 with the staged strategy, because this is our best-performing model in the benchmark. It achieved 93.48 percent test accuracy and 0.9311 macro F1 on Stanford Dogs."

### Optional comparison line
"If needed, we can switch immediately to ResNet-50 and compare the prediction behavior between a CNN backbone and a Transformer backbone."

---

## Demo Step D. Upload an image

### What to do on screen
- Use `Upload a dog image`
- Or paste an image from clipboard if that is faster

### What to say
"Now I upload a dog image. The app accepts a standard image file and prepares it using the same preprocessing pipeline used in training, so the demo remains consistent with the benchmark setup."

### Good demo image tips
- clear dog photo
- single dominant dog in frame
- limited cluttered background
- breed that is visually distinctive

---

## Demo Step E. Run prediction

### What to do on screen
- Click `Predict & Explain`

### What to say while waiting
"When I click Predict and Explain, the app runs inference and also generates an interpretability visualization, so we can see not only the final class prediction but also an approximate view of where the model concentrates information."

---

## Demo Step F. Explain the prediction output

### What to do on screen
- Point to:
  - `Predicted breed`
  - `Top-1 confidence`
  - `Top predictions`

### What to say
"The first output is the predicted breed. Next, the top-1 confidence tells us how confident the model is about that prediction. Below that, the app shows the top candidate classes. This is useful because Stanford Dogs is a fine-grained dataset, so visually similar breeds may appear close together even when the final prediction is correct."

### If the prediction is correct
"In this example, the model predicts the correct breed with high confidence, and the top alternatives are also semantically similar breeds, which is a reasonable behavior for a fine-grained benchmark."

### If the prediction is wrong
"In this case, the prediction is not exactly correct, but the top alternatives are still visually similar breeds. This kind of error is expected in fine-grained classification and is one reason why we analyze more than just raw accuracy."

---

## Demo Step G. Explain the interpretability view

### What to do on screen
- Scroll to the explanation image
- If using ResNet-50, mention Grad-CAM
- If using ViT-B/16, mention attention rollout

### What to say for ViT-B/16
"Because I selected ViT-B/16, the app shows an attention-rollout visualization. This is more informative than simply displaying the last-layer attention map, because it summarizes how information flows across multiple Transformer blocks. We interpret it as an approximate indication of which spatial regions support the prediction, not as a perfect proof of causal importance."

### Extra explanation for the colors
"In the attention-rollout panel, warmer colors such as yellow, orange, and red indicate higher rollout values, meaning those regions receive stronger aggregated attention flow through the Transformer layers. Cooler colors such as blue or purple indicate lower values. In the overlay panel, the same color map is projected back onto the original image, so red or yellow regions are the areas the model is relatively emphasizing more, while blue regions contribute less in this visualization."

### Important clarification
"However, higher intensity here does not mean absolute certainty, and it does not prove that the model uses only that region. It is better understood as a qualitative importance map rather than an exact causal explanation."

### What to say for ResNet-50
"If we switch to ResNet-50, the app shows a Grad-CAM explanation. This highlights the local image regions that most influenced the classifier. It is a useful way to confirm whether the CNN is using meaningful dog-specific features."

### Safe interpretation line
"So for the demo, the main purpose of this panel is qualitative inspection. If the highlighted area overlaps the dog region, that is a good sign, but we still treat it as supportive evidence rather than a definitive explanation."

---

## Demo Step H. Mention calibration artifact

### What to do on screen
- Open the `Calibration artifact` expander

### What to say
"The app also includes a calibration artifact from the benchmark. This connects the live demo to our offline evaluation, where we measured not only accuracy and F1 but also calibration. In our experiments, the ViT staged model was better calibrated than the ResNet-50 staged model."

---

## 3. Suggested Closing Lines After the Demo

### Version 1
"So this demo shows that our project is not only a training benchmark, but also a usable interactive application where we can test different backbones, compare prediction behavior, and inspect interpretability in real time."

### Version 2
"To summarize the demo, the deployed app reflects the same four-model benchmark from our report, and it helps us verify that the best model, ViT-B/16 staged, performs strongly not only in metrics but also in practical interactive testing."

---

## 4. Fast Backup Script If Time Is Short

### 20-second version
"Here is our Streamlit demo for the image track. It exposes all four trained checkpoints from the benchmark. I select the best model, ViT-B/16 staged, upload one dog image, and run prediction. The app returns the predicted breed, confidence, top candidate classes, and an attention-rollout visualization, so we can evaluate both performance and model behavior in a practical setting."

---

## 5. Safety Notes for Live Presentation

- Prepare 2 or 3 dog images in advance.
- Keep one easy image and one challenging image.
- Default to `ViT-B/16 · Head 3 + Full 8`.
- If loading is slow, say:
  - "The app is loading the checkpoint from cloud storage, so the first request may take a few more seconds."
- If prediction is wrong, frame it positively:
  - "This is still useful because it shows the difficulty of fine-grained classification and lets us inspect the model's reasoning."
