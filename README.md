# Amex_AnalyzeThis
Neural networks on AnalyzeThis 2016 competition data 


<h2> Background </h2>
From the website: 
The Island of Hoenn is gearing up for upcoming polls. Citizens are waiting with bated breath as news agencies reveal their predictions on which party is likely to emerge victorious.
Much to the disappointment of all the citizens, there are discrepancies in these poll predictions amongst news channels.
So many inconsistent predictions did not go down well with an inquisitive bunch of students. Wondering how difficult it might be to crack it, comes their 'Eureka' moment.
An idea to create their own ‘Start Up’ to analyze poll sentiments and predict the winner. A start up called - Analyze This.
They gather data for a sample of citizens of Island of Hoenn and get started.
Information on historical voting pattern, rally attendance and demographics is what they have at hand to predict the winner amongst the 5 competing parties.

Can you help these students crack this puzzle? Do you have it in you to start your own Analyze This?


<h2> Problem Statement </h2>

You have to predict the party for which each citizen will vote for in the upcoming polls.
<br><br>
Source: https://in.axpcampus.com/AnalyzeThis/campusactivity/problem-statement.php


<h2> Data for Analysis </h2>

Following files can be downloaded for your analysis. <br><br>
1. Training_Dataset.csv: This data has all the information for a sample of citizens. The information includes:
<br>
&nbsp; &nbsp; a. Past history of voting for all citizens
<br>
&nbsp; &nbsp; b. Who they will vote for in upcoming polls
<br>
&nbsp; &nbsp; c. Donation, Rally Attendance
<br>
&nbsp; &nbsp; d. Demographics of these citizens
<br><br>
2. Leaderboard_Dataset.csv: This data has information on the historical vote for a different set of citizens, along with donation, rally attendance and demographics. The vote in the upcoming polls is not present in the data.
<br><br>
3. Final_Dataset.csv: This data has information on the historical vote for a different set of citizens, along with donation, rally attendance and demographics. The vote in the upcoming polls is not present in the data.
<br><br>
4. Data_Dictionary.xlsx: This sheet will give you the description of all the variables contained in the 3 datasets above.
<br><br>
Please note that you can make multiple submissions corresponding to the Leaderboard Dataset. However, for the Final dataset you can submit only one solution. For further details, please refer to the submission guidelines document available at the link below: http://in.axpcampus.com/AnalyzeThis/campusactivity/guidelines-and-submission.php
<br><br>


<h2>Evaluation Criteria </h2>

Leader board Submission
<br>
The dataset for Leader board evaluation would be evaluated on the basis of the estimation that you provide. The score is calculated as:
<br>
&nbsp; &nbsp; a. If Actual Vote = Predicted Vote and Actual Vote = Historical Vote, then score = 50
<br>
&nbsp; &nbsp; b. If Actual Vote = Predicted Vote and Actual Vote ^= Historical Vote, then score = 100
<br>
&nbsp; &nbsp; c. If Actual Vote ^= Predicted Vote and Actual Vote = Historical Vote, then score = 0
<br>
&nbsp; &nbsp; d. If Actual Vote ^= Predicted Vote and Actual Vote ^= Historical Vote, then score = -50
<br>
