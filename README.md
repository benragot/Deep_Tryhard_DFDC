# DeepFake Detection Challenge (DFDC) by Ulysse, Jean-Baptiste, Christophe and Benjamin.
> The DFDC is a Kaggle challenge that was organised two years ago.
> The link here :arrow_right: [Kaggle DFDC](https://www.kaggle.com/c/deepfake-detection-challenge)

# Our approach
> You can find our approach on this google slides :arrow_right: [Our slides](https://docs.google.com/presentation/d/1HIfiFZGQcne0gnbyxlqfHt6dCbZ-ZzzyAyGkfhgy81c/edit?usp=sharing)
> Basically, we built a face classifier based on a Convolutionnal Neural Network. It can predict wether a face is DeepFake or Real.
> Therefore, we use this classifier on 10 faces extracted form a video to predict whether the whole video is DeepFake or Real.
## The data
> The dataset was 500 GB of videos labeled as DeepFake or Real.
> It represents 120 000 videos of 10s each.
## Our selection
> The dataset was heavily unbalanced (80% of Fake) so we decided to build a 50/50 dataset of 38 000 videos.
> Then we collected 100 frame out of the 300 (30 FPS) of each videos
> For each frame, we collected one face and resized it in 224x224.
> So we had 100 faces per video, and we decided to select three of them showing the greatest wariance, in order to have a great variety for our train set.
## The model
> We then trained different CNNs on these faces to classify wether they are DeepFake or Real.
> After some research, we ended up with the architecture that you can find in models/model_T2PV_KERsize_3_MP_4_Denses_128_16_drop_10/model_summary.txt
> You can find in models/ plenty of information in the model_T2PV_KERsize_3_MP_4_Denses_128_16_drop_10 folder.
> These informations were automatically generated by our module turbomodel_trainer.py
