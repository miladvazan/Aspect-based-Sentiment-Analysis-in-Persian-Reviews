# Aspect-based-Sentiment-Analysis-in-Persian-Reviews
codes for our paper [Jointly Modeling Aspect and Polarity for Aspect-based Sentiment Analysis in Persian Reviews](https://arxiv.org/abs/2109.07680)

# Jointly Modeling Aspect and Polarity for Aspect-based Sentiment Analysis in Persian Reviews
Identification of user's opinions from natural language text has become an exciting field of research due to its growing applications in the real world. The research field is known as sentiment analysis and classification, where aspect category detection (ACD) and aspect category polarity (ACP) are two important sub-tasks of aspect-based sentiment analysis. The goal in ACD is to specify which aspect of the entity comes up in opinion while ACP aims to specify the polarity of each aspect category from the ACD task. The previous works mostly propose separate solutions for these two sub-tasks. This paper focuses on the ACD and ACP sub-tasks to solve both problems simultaneously. The proposed method carries out multi-label classification where four different deep models were employed and comparatively evaluated to examine their performance. A dataset of Persian reviews was collected from CinemaTicket website including 2200 samples from 14 categories. The developed models were evaluated using the collected dataset in terms of example-based and label-based metrics. The results indicate the high applicability and preference of the CNN and GRU models in comparison to LSTM and Bi-LSTM. 

## About dataset
Details of the dataset collected from user’s comments in the field of movie scoring from the
CinemaTicket website 
![](data.jpg)

#### dataset
If you need the dataset of this article, email this address:
vazanmilad@gmail.com
## Citation

    @misc{vazan2021jointly,
        title={Jointly Modeling Aspect and Polarity for Aspect-based Sentiment Analysis in Persian Reviews},
        author={Milad Vazan and Jafar Razmara},
        year={2021},
        eprint={2109.07680},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }
