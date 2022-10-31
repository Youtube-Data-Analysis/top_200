# <a name="top"></a>Predicting Github Repository Programming Language based on project README files
![]()

by: Vincent Banuelos and J. Vincent Shorter-Ivey

***
[[Project Description/Goals](#project_description_goals)]
[[Project Description/Goals](#business_goals)]
[[Research Questions](#research_questions)]
[[Planning](#planning)]
[[Data Dictionary](#dictionary)]
[[Reproduction Requirements](#reproduce)]
[[Pipeline Takeaways](#pipeline)]
[[Conclusion](#conclusion)]

___

## <a name="project_description_goals"></a>Project Description/Goals:
- Perform analysis on trending Youtube videos and the channels that create them in order to craft a machine learning algorithm predicting a videos placement in Top 25 of Youtubes Top 200 trending list. 

- This project runs through the entire Data Science Pipeline and culminates with classification modeling techniques.

- Utilizes the Top 1

## <a name="business_goals"></a>Project Description/Goals:
- To aid content creators in creating content that has greater possibility of becoming a top trending video

- To increase content outreach and audienced relevance

- More efficient audience growth for creators without large audiences

[[Back to top](#top)]


## <a name="research_questions"></a>Research Questions:

- What are the most common words in the READMEs?
  - 

- Does the length of the README vary by programming language?
  - 

- Do different programming languages use a different number of unique words?
  - Yes, each programming language has a different number of Unique words.
    - Python: 1733
    - JavaScript: 219
    - HTML: 83
    - Other: 607

- Are there any words that uniquely identify a programming language?
  - 

[[Back to top](#top)]


## <a name="planning"></a>Planning:

- Create README.md with data dictionary, project goals, and come up with initial hypotheses.
- Acquire data from single repository as test. After succesful test, decide on 100 repositories to analyze.
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

[[Back to top](#top)]

## <a name="dictionary"></a>Data Dictionary  

| Target Attribute | Definition | Data Type |
| ----- | ----- | ----- |
|langauge|the langauge of the repository|object|
---
| Feature | Definition | Data Type |
| ----- | ----- | ----- 
| word_count | count of README words  |float64 
| lemmatized | ----- |float64/object
| language_bigrams | word bigrams popular within each language | float64/object
---

## <a name="reproduce"></a>Reproduction Requirements:

You will need your own ******** then follow the steps below:

  - Download the csv files, wrangle.py, model.py, explore.py, and final_report.ipynb files
  - Run the final_report.ipynb notebook

[[Back to top](#top)]


## <a name="pipeline"></a>Pipeline Conclusions and Takeaways:

###  Wrangling Takeaways
- 
---
### Nulls/Missing Values
* 
---
### Feature Engineering 
* 
---   
### Flattening
* 
---

### Exploration Summary

- 

# Modeling
- 
--- 

[[Back to top](#top)]


## <a name="conclusion"></a>Conclusion, and Next Steps:

# Conclusion
## Summary of Key Findings
* 
* 
* 
* 
---
## Suggestions and Next Steps
* 
[[Back to top](#top)]