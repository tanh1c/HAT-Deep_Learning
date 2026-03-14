## HO CHI MINH CITY UNIVERSITY OF

## TECHNOLOGY

### (HCMUT) – VNUHCM

### Faculty of Computer Science and Engineering

# Assignment 1

## Deep Learning and Its Applications

### Course code: CO

```
Academic year & Semester: 2025–2026, Semester 2
```
```
Instructor: Le Thanh Sach
```
### Assignment handed out to students


## Contents

- 1 Topic and format
   - 1.1 Topic
   - 1.2 Format
- 2 Learning objectives
- 3 Dataset requirements
   - 3.1 Constraints (to avoid overly simple datasets)
- 4 Technical requirements
   - 4.1 Image dataset
   - 4.2 Text dataset
   - 4.3 Multimodal dataset
   - 4.4 Evaluation metrics
- 5 Grading criteria
- 6 Report and deliverables
   - 6.1 Landing page (GitHub Pages)
      - 6.1.1 Common page (home)
      - 6.1.2 Page for each assignment
- 7 Submission and deadlines
   - 7.1 Submission requirements
   - 7.2 Deadlines


## 1 Topic and format

### 1.1 Topic

Classificationon three types of data: images, text, and multimodal (image + text).

### 1.2 Format

- Work ingroups of 3–4 students.
- Students register their group via the link published on the LMS.
- Note: The registered group will work together onallassignments in the course, not
    only Assignment 1.

## 2 Learning objectives

Upon completing this assignment, students will be able to:

- Apply pretrained models (CNN, ViT, RNN, Transformer, multimodal models) to clas-
    sification on image, text, and multimodal data.
- Prepare data (Dataset, DataLoader), apply appropriate augmentation, and set up train-
    ing/evaluation pipelines.
- Compare and analyze results across model families (CNN vs. ViT, RNN vs. Trans-
    former, zero-shot vs. few-shot) using tables and figures.
- Present and defend results via report, demo video, and presentation video; organize
    code and documentation on GitHub Pages.
- (Optional) Extend with interpretability techniques or other approaches to improve the
    grade.

## 3 Dataset requirements

Each group must useall threeof the following dataset types (self-selected or as suggested
by the instructor):

1. Image datasetfor classification (image classification).
2. Text datasetfor classification (text classification).
3. Multimodal datasetfor classification (multimodal: image + text).


### 3.1 Constraints (to avoid overly simple datasets)

Datasets must meet at least the following; otherwise, the work may be marked down or
the group may be asked to change the dataset:

- Number of classes:at least5 classesper dataset (image, text, multimodal). Binary
    or 3–4 class problems are not sufficient to demonstrate model comparison.
- Size:training set of at leastseveral thousand samples(e.g.≥5 000for image/text).
    Very small datasets (a few hundred samples) are not sufficient to evaluate CNN/ViT
    or RNN/Transformer convincingly.
- Difficulty: prefer datasets with moderate or higher class separability. For images:
    avoid using only MNIST; consider CIFAR-10/100, Fashion-MNIST. For text: avoid
    very short or trivial sets (e.g. 2-class sentiment on very short sentences); aim for varied
    length and semantics.
- Multimodal:the multimodal dataset must havegenuine image–text pairs(describ-
    ing the same entity/event), not randomly paired images and text. Examples: COCO
    captions, Flickr30k, or equivalent.

Groups should state in the report why each dataset was chosen and how the above con-
straints are satisfied. If unsure about dataset validity, consult the instructor before im-
plementation.

## 4 Technical requirements

### 4.1 Image dataset

Compare models fromtwo families:

- CNN(Convolutional Neural Networks)
- ViT(Vision Transformer)

(Use pretrained models, train/fine-tune and evaluate; present a comparison of results.)

### 4.2 Text dataset

Compare models fromtwo families:

- RNN(Recurrent Neural Networks, e.g. LSTM)
- Transformer


### 4.3 Multimodal dataset

Comparetwo approaches:

- Zero-shot classification
- Few-shot classification

### 4.4 Evaluation metrics

Groups must report at leastaccuracy(and F1when classes are imbalanced); use the
same metrics across models for fair comparison. Additional metrics (precision, recall,
confusion matrix) may be included if appropriate.

## 5 Grading criteria

- 60% of grade:Results from using pretrained models, training (fine-tuning), and eval-
    uation on all three data types (image, text, multimodal) according to the comparison
    requirements in Section 4.
- 40% of grade:Fromextensionschosen by the group. Suggestions:
    - Interpretability: which regions of the image or text the model “looked at” for
       the classification decision (e.g. attention visualization, saliency map, Grad-CAM,
       etc.).
    - Error analysis:categorizing errors (confusion, hard examples), illustrating a few
       misclassified cases with brief explanation.
    - Fine-tuning strategy comparison: freeze backbone vs full fine-tune vs layer-
       wise learning rate; report accuracy and training time.
    - Augmentation & robustness:compare with/without augmentation (RandAug-
       ment, MixUp, CutMix for images; back-translation for text); or test robustness to
       light noise/corruption.
    - Model efficiency: compare accuracy vs model size / inference time; (optional)
       simple compression: pruning, quantization, or smaller models (MobileNet, Distil-
       BERT).
    - Ensemble:combine at least two models (e.g. CNN + ViT, RNN + Transformer);
       compare single model vs ensemble.
    - Application demo:simple interface (Gradio, Streamlit, Flask) for users to try;
       optionally with visual explanation.


- Calibration: assess prediction confidence (ECE, reliability diagram); compare
    which model is “confidently correct” more often.
- Imbalanced data:if the dataset has class imbalance, try reweighting, oversam-
    pling, or focal loss; report per-class metrics.
- Other directions: groups may propose other extensions (state clearly in the
    report and may discuss with the instructor).

## 6 Report and deliverables

### 6.1 Landing page (GitHub Pages)

Groups must createone landing page on GitHub Pagesto presentallassignments
in the course (including Assignment 1).

#### 6.1.1 Common page (home)

The common page must contain:

- Group name
- Names of members
- Course instructor name
- Links to each assignment in the course (including Assignment 1)

#### 6.1.2 Page for each assignment

Foreachassignment (including Assignment 1), there must be a dedicated page containing:

- Assignment name
- (Briefly) Member names and instructor
- Link to thedemo videofor this assignment
- Link to thepresentation video (YouTube)for this assignment. Note: the instructor
    will watch the video during grading if groups do not present in person.
- Link to thecodefor this assignment
- Link to thepresentation contentfor this assignment, including at least:
    1. Report on problem and dataset exploration (EDA)
    2. Report on Dataset, DataLoader, and Augmentation setup


3. Report on model building, training, evaluation, and comparison
4. Experimental results report: tables; figures; analysis and discussion
5. Other extension reports (if any)

## 7 Submission and deadlines

### 7.1 Submission requirements

In addition to the content published on GitHub Pages, each group must submitpresen-
tation slidessummarizing the work (following the format and submission path specified
on the LMS). Reports are submitted in two rounds as below. The presentation grade
is based on Report 1 and the final report according to the weights given in the Dead-
lines subsection. The instructor may ask groups to present at various stages during the
assignment.

### 7.2 Deadlines

- Report 1(50% of presentation grade): 23:59, 26 March 2026.
- Final report(100% of presentation grade): 23:59, 06 April 2026.
- Late submission: Each week late after the corresponding deadline incurs a 20%
    deduction on the presentation grade for that round.

Submission links and detailed rules (file format, file names) are published on the LMS.


