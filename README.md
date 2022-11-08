# <a name="top"></a>Anlayzing the Top 200 Trending Video List
![]()

by: Vincent Banuelos, Cristian Ibarra, Patrick Amwayi, and Shorter

***
[[Project Description/Goals](#project_description_goals)]
[[Business Goals](#business_goals)]
[[Research Questions](#research_questions)]
[[Planning](#planning)]
[[Data Dictionary](#dictionary)]
[[Reproduction Requirements](#reproduce)]
[[Pipeline Takeaways](#pipeline)]
[[Conclusion](#conclusion)]

___

## <a name="project_description_goals"></a>Project Description/Goals:
- Perform analysis on trending Youtube videos and the channels that create them in order to craft a machine learning algorithm predicting a videos placement in Top 25 of Youtube's Top 200 trending list. 

- This project runs through the entire Data Science Pipeline and culminates with classification modeling techniques.

## <a name="business_goals"></a>Project Description/Goals:
- To aid content creators in creating content that has greater possibility of becoming a top trending video

- To increase content outreach and audience relevance

- More efficient audience growth for creators without large audiences

[[Back to top](#top)]


## <a name="research_questions"></a>Research Questions:

* Are videos with disabled comments obtaining more views the none disable comments?
  
* Does comment value and views have a correlation together?
   
* What is the most frequent occuring bigrams per category?
 
* Does category effect total amount of words?

[[Back to top](#top)]


## <a name="planning"></a>Planning:
- Phase One
  - Create README.md with data dictionary, project goals, and come up with initial hypotheses.
  - Acquire data from single country as test. After succesful test, decide on which other countries to pull data about, and acquire.
  - Clean and prepare data for the first iteration through the pipeline, MVP preparation. Create a functions to automate the process. 
  - Store the acquisition and preparation functions in a wrangle.py module function, and prepare data in Final Report Notebook by importing and using the function.
  - Clearly define at least **four** hypotheses, set an alpha, run the statistical tests needed, reject or fail to reject the Null Hypothesis, and document findings and takeaways.
  - Establish a baseline accuracy and document well.
  - Train at least **five** different classification models.
  - Evaluate models on train and validate datasets.
  - Choose the model that performs the best and evaluate that single model on the test dataset.
  - Document conclusions, takeaways, and next steps in the Final Report Notebook.
  - Prepare slideshow and recorded presentation for classmate and faculty consumption
    - Slide deck available *here <insert URL to slide deck>*
- Phase Two of Project 
  - Decide on focus for expanded data acquisition relating to Youtube Channel content. 
  - Incorporate data into existing streams
  - Adjust conclusions, presentation, findings, and suggestions as necessary to accomodate new information. 

[[Back to top](#top)]

## <a name="dictionary"></a>Data Dictionary  

| Column (*engineered)  |Description     | Count    | Dtype      
:-----------------------|:---------------|:---------|:-----------
------                  | -------------- | -----    | ---    
video_id                |idenfitification code                      | 2019 | object     
title                   |title of video                             | 2019 | object     
publishedAt             |datetieme of publishing                    | 2019 | datetime64 
channelTitle            |title of publishing channel                | 2019 | object     
categoryId              |category designation                       | 2019 | object     
trending_date           |trending date                              | 2019 | datetime64 
tags                    |video tags                                 | 2019 | object     
view_count              |number of video views                      | 2019 | int64      
likes                   |number of video likes                      | 2019 | int64      
comment_count           |number of video comments                   | 2019 | int64      
thumbnail_link          |status of video thumbnail                  | 2019 | object     
comments_disabled       |status of video comments                   | 2019 | bool       
ratings_disabled        |status of video ratings                    | 2019 | bool       
description             |video description                          | 2019 | object     
duration                |video duration                             | 2019 | int64      
captions                |status of video captions                   | 2019 | bool       
region                  |region video appears on list               | 2019 | object     
rank                    |higest rank on trending list               | 2019 | int64      
top_25                  |status in top 25 on trending list          | 2019 | int64      
channel_age             |age of channel                             | 2019 | int64      
subscribers             |number of channel subscribers              | 2019 | int64      
video_count             |number of videos published by channel      | 2019 | int64      
age                     |age of video                               | 2019 | float64    
engagement              |level of video engagement                  | 2019 | float64    
sponsored               |status of video sponsorship                | 2019 | int64      
num_of_tags             |number of tags used for video              | 2019 | int64      
word_bank               |list of words used in description          | 2019 | object     
cleaned_tags            |set of tags used                           | 2019 | object     
cleaned_desc            |set of words used in description           | 2019 | object     
title_in_description    |status of title in description             | 2019 | int64      
title_in_tags           |status of title in tags                    | 2019 | int64      
pct_tags_in_description |percent of tags appearing in description   | 2019 | float64    
title_lengths           |length of title                            | 2019 | int64      
desc_lengths            |length of description                      | 2019 | int64      
tags_length             |length of tags                             | 2019 | int64      
views_per_sub           |video views per channel subscriber         | 2019 | float64    
content_rate            |rate of channel content release            | 2019 | float64    

---

## <a name="reproduce"></a>Reproduction Requirements:

* Utilize the linked .pkl file that contains the data for the project
  * https://drive.google.com/file/d/1twg97V0zm_OUcnyWqzYRs38iNuI-5Yc1/view?usp=share_link
  * Alternatively, you will need your own Youtube Analytics API key *** to acquire new data
* Follow the steps below:
  * Download prepared.pkl, acquire.py, model.py, explore.py, and final_report.ipynb files
    * Alternate: run scraper.py with your Youtube API credentials and download the above files.
  * Run the final_report.ipynb notebook

[[Back to top](#top)]


## <a name="pipeline"></a>Pipeline Conclusions and Takeaways:

##  Wrangling Takeaways
* ** VINCENT ** Insert
* Feature creation was large workload as scraping provided just basic info
* Had to generate features for further surface level insight
* Alot of time spent converting date into useable formats for exploration
  * Example: `duration` being a string

### Nulls/Missing Values
* Nulls found in `descriptions` were handled by filling with 'No Description"

### Feature Engineering (Highlights)
* Top 25 (TARGET): Has the video ever been ranked in the top 25
* Rank: Highest rank acheived by a video worldwide
* Engagement: The level of audience engagement
* Word Bank: Words in the tags and descriptions
* Age: How old was a video at its peak rank

---

## Exploration Summary
- Entertainment might have been the highest overall but doesnt mean it was liked/or viewed the most. Just means most videos are categories as entertainment
- Music/Entertainment have the highest words count but it could be because they both take up at least 15 percent and above of the whole dataframe
- Region effect on trending video alot because in some country music isnt the highest or entertainment 
- Having the comments enabled doesnt really effect the like/views to high so i think it doesnt matter if the video is bad or good as longest it gets views or some likes.
---

## Modeling Summary
- All models had roughly almost the same accuracy for train and validate sets.
- Logistic Regression model performed the worst on out-of-sample data.
- The best performing model is Decision Tree Classifier.
- However, from the above model analysis we can see that the highest accuracy is achieved by Decision Tree. It performs better than baseline by about 7%.

- While this is an improvement there is still room for improvement in future iterations
--- 

[[Back to top](#top)]


## <a name="conclusion"></a>Conclusion, and Next Steps:

## Summary of Key Findings
* Entertaiment videos are by far the most popular
* Non-Profit videos receive the most engagement
* Global trends stay fairly consistent, but their is some variance
* Release timing is an important factor when wanting to make the jump to becoming a Top 25 video
---
## Suggestions and Next Steps
* Recommendations: Focus content on a mix of Non-Profit and Entertainment content released early in the day to capture widest audience
* Future Talk: With more time we would incorporate more data featuring channel analytics because who is posting is just as important as what is being posted
* Next Steps: Work to capture more channel information. Provide deeper insights in regards to content release schedule and global appeal

[[Back to top](#top)]