## Periodontitis-Classification

Deep learning models have proven their efficiency in performing various tasks in the fields of object detection, image classification, facial detection, vehicle recognition, speech recognition, and diabetic retinopathy (Indolia et al., 2018). Specifically, the convolutional neural network (CNN) is a common deep learning model used in the classification and detection of various dentistry-related diseases through X-ray image analysis (Hiraiwa et al., 2019; Miki et al., 2017; Murata et al., 2019). CNNs work directly on the input images and generate the required output without demanding the execution of intermediate steps such as segmentation and feature extraction (Ghosh et al., 2020). However, designing and training deep neural networks are complex, time-consuming, and costly endeavors (Atas et al., 2022). Recent studies have developed various CNN-based approaches to automatically classify the stage of periodontitis of individual teeth (Kim et al., 2019; Krois et al., 2019; Lee et al., 2018). However, most of these studies rely on heavily parameterized models and hybrid deep learning frameworks that may impact the speed and execution of the model. Therefore, this project aims to focus on maximizing computational resource efficiency and speed by leveraging the benefits of one of the more recent transfer learning-based approaches, specifically MobileNets.

The experiments conducted in this project were carried out using the Python 3.9-based Keras framework with Tensorflow (Géron, 2019). The program was implemented using a Graphical Processing Unit (GPU) and 12 GB of RAM on Google Collaboratory (Bisong, 2019). The classification of dental panoramic radiographs was performed using MobileNetV2, thereafter, the results were evaluated. The experiment begins with the collection of a panoramic radiograph dataset from Kaggle an open-source data repository. A total of 100 images were obtained, with 50 labeled as periodontal and 50 labeled as non-periodontal. Data augmentation was applied to the dataset due to the small number of images in the dataset. After augmentation, a total of 6958 were obtained which were then preprocessed to ensure they had the right representation before feeding into the deep learning model. The preprocessed dataset was then split into 70%, 20%, and 10%. Based on the split, 70% of the dataset was used for training, 20% was used for validation and 10% was used for testing. The training and validation splits were fed into the deep learning model. The test set was then used to evaluate the performance of the model. 

Furthermore, the augmented dataset developed in this project was also used on CNN-based periodontitis classification models that were proposed in previous studies (Aberin and Goma, 2018; Alotaibi et al., 2022; Joo et al., 2019; Kim et al., 2019; Lee et al., 2018). This was done to compare the performance of existing models and the MobileNet V2 model that was used in this project.

### References

Aberin, S.T.A., Goma, J.C.D., 2018. Detecting Periodontal Disease Using Convolutional Neural Networks, in: 2018 IEEE 10th International Conference on Humanoid, Nanotechnology, Information Technology,Communication and Control, Environment and Management (HNICEM). Presented at the 2018 IEEE 10th International Conference on Humanoid, Nanotechnology, Information Technology,Communication and Control, Environment and Management (HNICEM), IEEE, Baguio City, Philippines, pp. 1–6. https://doi.org/10.1109/HNICEM.2018.8666389

Alotaibi, G., Awawdeh, M., Farook, F.F., Aljohani, M., Aldhafiri, R.M., Aldhoayan, M., 2022. Artificial intelligence (AI) diagnostic tools: utilizing a convolutional neural network (CNN) to assess periodontal bone level radiographically—a retrospective study. BMC Oral Health 22, 399. https://doi.org/10.1186/s12903-022-02436-3

Atas, I., Ozdemir, C., Atas, M., Dogan, Y., 2022. Forensic Dental Age Estimation Using Modified Deep Learning Neural Network.

Bisong, E., 2019. Google Colaboratory, in: Building Machine Learning and Deep Learning Models on Google Cloud Platform. Apress, Berkeley, CA, pp. 59–64. https://doi.org/10.1007/978-1-4842-4470-8_7

Géron, A., 2019. Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow: concepts, tools, and techniques to build intelligent systems, Second edition. ed. O’Reilly Media, Inc, Beijing [China] ; Sebastopol, CA.

Hiraiwa, T., Ariji, Y., Fukuda, M., Kise, Y., Nakata, K., Katsumata, A., Fujita, H., Ariji, E., 2019. A deep-learning artificial intelligence system for assessment of root morphology of the mandibular first molar on panoramic radiography. Dentomaxillofacial Radiol. 48, 20180218. https://doi.org/10.1259/dmfr.20180218

Indolia, S., Goswami, A.K., Mishra, S.P., Asopa, P., 2018. Conceptual Understanding of Convolutional Neural Network- A Deep Learning Approach. Procedia Comput. Sci. 132, 679–688. https://doi.org/10.1016/j.procs.2018.05.069

Joo, J., Jeong, S., Jin, H., Lee, U., Yoon, J.Y., Kim, S.C., 2019. Periodontal Disease Detection Using Convolutional Neural Networks, in: 2019 International Conference on Artificial Intelligence in Information and Communication (ICAIIC). Presented at the 2019 International Conference on Artificial Intelligence in Information and Communication (ICAIIC), IEEE, Okinawa, Japan, pp. 360–362. https://doi.org/10.1109/ICAIIC.2019.8669021

Kim, J., Lee, H.-S., Song, I.-S., Jung, K.-H., 2019. DeNTNet: Deep Neural Transfer Network for the detection of periodontal bone loss using panoramic dental radiographs. Sci. Rep. 9, 17615. https://doi.org/10.1038/s41598-019-53758-2

Krois, J., Ekert, T., Meinhold, L., Golla, T., Kharbot, B., Wittemeier, A., Dörfer, C., Schwendicke, F., 2019. Deep Learning for the Radiographic Detection of Periodontal Bone Loss. Sci. Rep. 9, 8495. https://doi.org/10.1038/s41598-019-44839-3

Lee, J.-H., Kim, D., Jeong, S.-N., Choi, S.-H., 2018. Diagnosis and prediction of periodontally compromised teeth using a deep learning-based convolutional neural network algorithm. J. Periodontal Implant Sci. 48, 114. https://doi.org/10.5051/jpis.2018.48.2.114

Miki, Y., Muramatsu, C., Hayashi, T., Zhou, X., Hara, T., Katsumata, A., Fujita, H., 2017. Classification of teeth in cone-beam CT using deep convolutional neural network. Comput. Biol. Med. 80, 24–29. https://doi.org/10.1016/j.compbiomed.2016.11.003

Murata, M., Ariji, Y., Ohashi, Y., Kawai, T., Fukuda, M., Funakoshi, T., Kise, Y., Nozawa, M., Katsumata, A., Fujita, H., Ariji, E., 2019. Deep-learning classification using convolutional neural network for evaluation of maxillary sinusitis on panoramic radiography. Oral Radiol. 35, 301–307. https://doi.org/10.1007/s11282-018-0363-7

