The following files are part of the Bias in Advertising data hosted on IBM Developer Data Asset eXchange.

Homepage: https://developer.ibm.com/exchanges/data/all/bias-in-advertising/
Download link: https://dax-cdn.cdn.appdomain.cloud/dax-bias-in-advertising/1.0.0/bias-in-advertising.tar.gz

File list: 
-- add_campaign_dats.csv : the core dataset.
-- LICENSE.txt : a plaintext version of the Community Data License Agreement - Permissive - Version 2.0 license that the dataset is licensed under. (https://cdla.dev/permissive-2-0/)

Dataset Overview:
* To demonstrate discovery, measurement, and mitigation of bias in advertising, we use a dataset that contains synthetic generated data of all users who were shown a certain advertisement. Each instance of the dataset has feature attributes such as gender, age, income, political/religious affiliation, parental status, home ownership, area (rural/urban), and education status.
 
* The predicted probability of conversion along with the binary predicted conversion, which is obtained by thresholding the predicted probability, is included. In addition, the binary true conversion, based on whether the user actually clicked on the ad is also included. 

* A user is considered to have converted (true conversion=1) if they clicked on the ad. 

* This data is typically gathered using an ad targeting platform, where dynamic creative optimization algorithms target users who are more likely to click on the ad creative (more likely to convert). 

* Targeting involves choosing user specific attributes such as the market area, age group, income etc. and showing the particular ad to those users who have these attributes.

* After the ad campaign is completed, the ad data is obtained from various ad data management platforms.
 
* They are used with the AIF360 toolkit to: 
   1. discover the subgroups that exhibit high predictive bias using the multidimensional subset scan (MDSS) method, 
   2. measure the bias exhibited by these subgroups using various metrics, and 
   3. mitigate bias using post-processing bias mitigation approaches. 

* MDSS is a technique used to identify the subset of attributes and corresponding feature values (aka subgroups) that have the most predictive bias. The group with the highest predictive bias is designated as "privileged", and the rest of the records belong to "unprivileged" groups in our analysis. 

* Note that the use of terms privileged and unprivileged in this analysis does not correspond to socioeconomic privilege.

