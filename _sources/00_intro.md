---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.2
kernelspec:
  display_name: appinsight
  language: python
  name: python3
---

# AppVoC Dataset: Apple App Store Voice of the Customer at Scale

> "Because we offer nearly two million apps — and we want you to feel good about using every single one of them." - Apple

## Introduction
In 1983, Steve Jobs envisioned a future where consumers could seamlessly purchase and download software directly from their computers. Fast forward 25 years to 2008, Apple launched the App Store, realizing Jobs’ vision and revolutionizing the way we interact with technology. Following the success of the iTunes digital music store, the Apple App Store became one of the world’s first commercially successful mobile app marketplaces, and today, it stands as a digital leviathan. As of May 29, 2024, the App Store boasts a staggering 1,928,363 apps available for download, catering to over 1.3 billion iOS users worldwide.

And, amidst this flourishing ecosystem, understanding how users *actually* feel about the apps they download and use every day, remains crucial. It's incontestable. Ratings and reviews play a pivotal role in App Store Optimization (ASO) and app discoverability, which can be the difference between app success and failure. Yet, the dearth of large-scale customer rating and review datasets for in-depth analysis, astounding.

Disaffected by the scarcity of comprehensive App Store review datasets, an undertaking was begun to construct a collection of app reviews that would serve as a laboratory where the frontier of generative methods, text synthesis, graph neural networks, next-gen NLP solutions, advanced integration technologies, and other innovative methods can be explored, stretched, taxed and charged with revealing the nuances and contours of the customer experience, to uncover the latent, and perhaps indistinguished emergent market need with AI.

Today, we bridge this gap with the introduction of the AppVoC (Voice of the Customer) dataset — a collection of iOS app reviews, believed to be one of the largest collections of Apple IOS user review datasets available, second only to that of the App Store itself.

### Key Features of AppVoC Dataset

- **Scale**: The dataset includes a total of 18,301,035 reviews, making it one of the most extensive collections of app store reviews publicly available.
- **User Base**: It covers 13,242,130 unique users, reflecting a vast and diverse range of perspectives.
- **App Diversity**: The dataset includes reviews for 34,051 unique apps across 11 categories, showcasing the broad spectrum of app interests and needs.
- **Category Coverage**: The dataset spans ten of the most popular app categories, illuminating the diverse interests and needs of mobile users.
- **Voice-on-Voice**: The dataset captures how users engage and value the sentiments and opinions of other users. The number and value of user votes on reviews opens up new avenues allowing researchers and analysts to explore user engagement dynamics, identify influential reviews, and understand the collective sentiment of the user community towards specific apps. These insights can inform app developers, stakeholders, and decision-makers in optimizing app experiences, addressing user concerns, and driving user engagement and satisfaction.
- **Temporal Coverage**: Spanning from 2008 to 2023, the dataset captures a wealth of user interactions and feedback over 15 years, providing a longitudinal view of app usage and user feedback trends. Timestamps illuminate the individual and collective progression of sentiment over time, motivating deeper exploration into the evolving dynamics of user engagement and feedback. By analyzing the temporal dimension of reviews, researchers and analysts can uncover trends, patterns, and fluctuations in user sentiment, shedding light on the factors influencing app popularity, user satisfaction, and overall app ecosystem dynamics. This temporal lens not only provides valuable insights into past user behavior but also enables predictive analysis and forecasting, empowering stakeholders to anticipate trends, mitigate risks, and capitalize on emerging opportunities in the ever-evolving app landscape.
- **Rich Metadata**: Each review is accompanied by metadata such as app name, category, rating, vote count, and timestamp, enabling multifaceted analysis.
- **Sentiment Analysis**: The dataset includes pre-computed sentiment scores for each review, aiding in the understanding of user satisfaction and feedback trends.

### Impact Beyond Research

While the AppVoC dataset holds immense value for academic research, its impact extends far beyond the confines of academia. For developers, stakeholders, entrepreneurs, data scientists, analysts, and professionals across the mobile app ecosystem, the AppVoC dataset isn't just a collection of reviews; it's a catalyst for innovation. Here's how:

#### For Developers:
Unlock the power of user feedback to create apps that resonate with your audience. For developers, AppVoC provides unparalleled insights into user sentiments, preferences, and behavior. Armed with this wealth of data, developers can fine-tune app features, address pain points, and create experiences that resonate with users on a profound level. Dive deep into user sentiments, preferences, and pain points to refine your app strategies, and prioritize feature enhancements and updates. Benchmark app performance against competitors and industry standards to stay ahead of the curve and deliver exceptional user experiences.

#### For Stakeholders:
Gain invaluable insights into user satisfaction, app performance, and market dynamics. By understanding user feedback and market trends, stakeholders can identify promising app opportunities, allocate resources strategically, and drive sustainable growth in an ever-evolving landscape. Monitor and manage brand reputation by addressing user concerns and enhancing user experiences. Make data-driven decisions that drive business growth, optimize app monetization strategies, and maximize ROI in the competitive landscape of the App Store.

#### For Data Scientists:
Data scientists unlock the true potential of AppVoC through advanced analytical techniques. From sentiment analysis to predictive modeling, AppVoC serves as a playground for data-driven innovation. By extracting actionable insights from the dataset, data scientists empower organizations to optimize app performance, enhance user experiences, and stay ahead of the curve. Conduct sentiment analysis to understand user emotions, opinions, and trends over time. Harness the richness of the AppVoC dataset to fuel your analytical endeavors. Leverage advanced machine learning and natural language processing techniques to extract actionable insights, conduct sentiment analysis, and uncover hidden patterns in user feedback. Develop personalized recommendation systems based on user preferences and behavior.

#### For Analysts:
Transform raw data into actionable intelligence with the AppVoC dataset. Track trends in app usage, and identify emerging market opportunities,  patterns, and shifts in user behavior to anticipate market changes. Track key performance indicators (KPIs) such as app ratings, user engagement, and retention rates for performance evaluation. Extract actionable insights to guide marketing strategies, user acquisition efforts, and competitive positioning. Empower decision-makers with the insights they need to drive success in the ever-evolving app ecosystem.

#### For Entrepreneurs and Business Owners:
Optimize monetization strategies through targeted advertising, in-app purchases, and subscription models. Identify niche markets and untapped opportunities for app expansion and diversification. Anticipate and mitigate potential risks by proactively addressing user concerns and feedback.

#### For Industry Professionals:
Whether you're a developer looking to create the next big app, a stakeholder striving to optimize app performance, or a data scientist seeking to extract insights from user feedback, we believe that this excursion into the AppVoC Dataset will provide tools, techniques, and inspiration needed to unlock the full potential of user data and drive meaningful innovations in the app ecosystem.

## AppVoC Overview
Enough with the preliminaries, let's dive in.

```{code-cell} ipython3
:tags: [remove-cell]

import os
if 'jbook' in os.getcwd():
    os.chdir(os.path.abspath(os.path.join("../..")))
import warnings
warnings.filterwarnings("ignore")
```

```{code-cell} ipython3
from myst_nb import glue
from appinsight.data.analysis.overview import DatasetOverview
from appinsight.infrastructure.storage.dataset import DatasetManager
```

```{code-cell} ipython3
dsm = DatasetManager()
df = dsm.read(directory="01_normalized",filename="reviews.pkl")
ov = DatasetOverview(data=df)
ov.overview
glue("overview", ov.overview)
```

As stated above and as indicated below, we have over 18 million reviews for 34,000 apss from some 13 million unique users.

The AppInsight Reviews dataset, extracted from the Apple App Store in July of 2023, represents user sentiments and experiences within the app ecosystem dating back to 2008. Comprising 18,301,035 reviews of approximately 34,000 mobile applications spanning 11 distinct categories, this dataset . The dataset encompasses 13 variables, offering a multifaceted view of user interactions and app attributes. Let's introduce the key variables encapsulated within the dataset:

- **id**: Unique identifier for each review.
- **app_id**: Unique identifier for the app being reviewed.
- **app_name**: Name of the mobile application being reviewed.
- **category_id**: Four-digit identifier representing the category or genre of the app.
- **category**: Category or genre name of the app.
- **author**: Name or identifier of the reviewer.
- **rating**: Numeric rating provided by the author for the app.
- **title**: Title of the review.
- **content**: Detailed content of the review provided by the author.
- **review_length**: Length of the review in words.
- **vote_sum**: Total sum of votes on the usefulness of the rating.
- **vote_count**: Number of votes on the usefulness of the rating.
- **date**: Date when the review was written.

These variables collectively evince the mobile app user experience and provide a base for analyzing user feedback and extracting actionable intelligence about areas for improvement, feature requests, unmet needs, and user preferences.

+++

# AppVOC Preprocessing
In this preprocessing overview, we introduce a structured approach comprising five stages to prepare review data for exploratory and interactive analysis:

1. **Data Quality Analysis**: Identify and rectify noise within the dataset, including profanity, excessive special characters, and identifiable patterns like emails and URLs.
2. **Cleaning**: Purge biased or distorted observations detected during the data quality analysis, ensuring data integrity.
3. **Feature Engineering**: Enhance data by transforming date fields into informative features such as month, day of the week, and day of the month. Additionally, anonymize author information to uphold privacy.
4. **Text Preprocessing**: Optimize textual data for downstream tasks such as word cloud generation and topic modeling, utilizing techniques like tokenization and stemming.
5. **Precomputation**: Establish an Analytics Precomputation Layer (APL) to precompute aggregations and statistical summaries, facilitating swift query responses and cost-effective analytics endeavors.

These stages lay the groundwork for subsequent in-depth exploration within the following sections. Each stage will be explored in detail, providing step-by-step insights to ensure a robust preprocessing foundation for our user feedback analytics.

+++

# Data Quality Assessment
The first stage of data processing is the Data Quality Assessment. This stage is crucial for ensuring that our dataset is ready for subsequent analysis and modeling tasks. By identifying and rectifying data quality issues early, we can avoid potential pitfalls that might compromise the integrity and accuracy of our results.

In this stage, we employ a series of tasks designed to identify and address any noise or irregularities within the dataset. Each task focuses on a specific aspect of data quality, ranging from detecting duplicate entries to identifying profanity, special patterns, and other potential sources of bias or distortion.

## Data Quality Checks
Now that are data are normalized, we proceed with a series of data quality checks to identify and address various issues:
1. **Duplicate Rows**: We identify and remove duplicate entries to ensure that each observation is unique, preventing skewed analyses and inflated metrics.
2. **Null Values**: We detect and handle missing data appropriately, which could involve imputation, deletion, or flagging incomplete records for further investigation.
3. **Outliers**: Check for outliers in numeric columns using the non-parametric Interquartile Range (IQR) method.
4. **Non-English Text**: We check for and address non-English text, as it may not be relevant to our analysis or could require special handling.
5. **Emojis**: Emojis can carry significant meaning in certain contexts but might also introduce noise. We identify and decide on their treatment—whether to retain, remove, or translate them into textual representations.
6. **Excessive Special Characters**: Special characters can disrupt text analysis and need to be managed, either by cleaning or encoding them appropriately.
7. **Invalid Dates**: We verify that date values fall within expected ranges and formats, correcting or flagging anomalies for further review.
8. **Invalid Ratings**: Ratings that fall outside the expected scale (e.g., 1 to 5) are identified and corrected or flagged.
9. **Profanity**: We detect and handle profane content to ensure that our dataset adheres to appropriate usage standards, especially if it's intended for public or sensitive applications.
10. **Special Patterns**: We identify and manage special patterns such as URLs, phone numbers, and emails. These patterns could be indicative of spam or need to be anonymized to protect privacy.

By conducting these data quality checks, we ensure that our dataset is clean, reliable, and ready for detailed analysis. This foundational step sets the stage for accurate insights and robust conclusions in the subsequent phases of our data processing pipeline.
