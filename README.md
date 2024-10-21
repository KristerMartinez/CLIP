# CLIP
CLIP (Contrastive Language–Image Pre-training) https://openai.com/index/clip/

We’re introducing a neural network called CLIP which efficiently learns visual concepts from natural language supervision. CLIP can be applied to any visual classification benchmark by simply providing the names of the visual categories to be recognized, similar to the “zero-shot” capabilities of GPT-2 and GPT-3.

Although deep learning has revolutionized computer vision, current approaches have several major problems: typical vision datasets are labor intensive and costly to create while teaching only a narrow set of visual concepts; standard vision models are good at one task and one task only, and require significant effort to adapt to a new task; and models that perform well on benchmarks have disappointingly poor performance on stress tests,1, 2, 3, 4 casting doubt on the entire deep learning approach to computer vision.

We present a neural network that aims to address these problems: it is trained on a wide variety of images with a wide variety of natural language supervision that’s abundantly available on the internet. By design, the network can be instructed in natural language to perform a great variety of classification benchmarks, without directly optimizing for the benchmark’s performance, similar to the “zero-shot⁠(opens in a new window)” capabilities of GPT-25 and GPT-3.6 This is a key change: by not directly optimizing for the benchmark, we show that it becomes much more representative: our system closes this “robustness gap” by up to 75% while matching the performance of the original ResNet-507 on ImageNet⁠(opens in a new window) zero-shot without using any of the original 1.28M labeled examples.

Key takeaways
1. CLIP is highly efficient

CLIP learns from unfiltered, highly varied, and highly noisy data, and is intended to be used in a zero-shot manner. We know from GPT-2 and 3 that models trained on such data can achieve compelling zero shot performance; however, such models require significant training compute. To reduce the needed compute, we focused on algorithmic ways to improve the training efficiency of our approach.

2. CLIP is flexible and general

Because they learn a wide range of visual concepts directly from natural language, CLIP models are significantly more flexible and general than existing ImageNet models. We find they are able to zero-shot perform many different tasks. To validate this we have measured CLIP’s zero-shot performance on over 30 different datasets including tasks such as fine-grained object classification, geo-localization, action recognition in videos, and OCR.B In particular, learning OCR is an example of an exciting behavior that does not occur in standard ImageNet models. Above, we visualize a random non-cherry picked prediction from each zero-shot classifier.

Limitations
While CLIP usually performs well on recognizing common objects, it struggles on more abstract or systematic tasks such as counting the number of objects in an image and on more complex tasks such as predicting how close the nearest car is in a photo. On these two datasets, zero-shot CLIP is only slightly better than random guessing. Zero-shot CLIP also struggles compared to task specific models on very fine-grained classification, such as telling the difference between car models, variants of aircraft, or flower species.

Broader impacts
CLIP allows people to design their own classifiers and removes the need for task-specific training data. The manner in which these classes are designed can heavily influence both model performance and model biases. For example, we find that when given a set of labels including Fairface39 race labelsC and a handful of egregious terms such as “criminal”, “animal,” etc., the model tends to classify images of people aged 0–20 in the egregious category at a rate of ~32.3%. However, when we add the class “child” to the list of possible classes, this behaviour drops to ~8.7%.

Additionally, given that CLIP does not need task-specific training data it can unlock certain niche tasks with greater ease. Some of these tasks may raise privacy or surveillance related risks and we explore this concern by studying the performance of CLIP on celebrity identification. CLIP has a top-1 accuracy of 59.2% for “in the wild” celebrity image classification when choosing from 100 candidates and a top-1 accuracy of 43.3% when choosing from 1000 possible choices. Although it’s noteworthy to achieve these results with task agnostic pre-training, this performance is not competitive when compared to widely available production level models. We further explore challenges that CLIP poses in our paper⁠(opens in a new window) and we hope that this work motivates future research on the characterization of the capabilities, shortcomings, and biases of such models. We are excited to engage with the research community on such questions.

Conclusion
With CLIP, we’ve tested whether task agnostic pre-training on internet scale natural language, which has powered a recent breakthrough in NLP, can also be leveraged to improve the performance of deep learning for other fields. We are excited by the results we’ve seen so far applying this approach to computer vision. Like the GPT family, CLIP learns a wide variety of tasks during pre-training which we demonstrate via zero-shot transfer. We are also encouraged by our findings on ImageNet that suggest zero-shot evaluation is a more representative measure of a model’s capability.

