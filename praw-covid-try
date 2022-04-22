import pandas as pd
from textwrap import dedent 
import apikeys
import praw
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ztest as ztest


redd = praw.Reddit(client_id = apikeys.RCLIENT_ID, 
                   client_secret = apikeys.RCLIENT_SECRET,
                   username = apikeys.RUSER,
                   password = apikeys.RPASSWORD,
    user_agent = f"MacOS:CommentScript:v1.1 (/u/{apikeys.RUSER})")
    
countries=['Ukraine','Estonia','Bulgaria','Moldova','Latvia','Iceland','Switzerland','Belgium','Bosnia and Herzegovina','Serbia','Portugal',
 'Luxembourg','Netherlands','Poland','Macedonia','Sweden','Malta','Croatia','Greece','Norway','Armenia','Ireland','Lithuania',
 'Spain','Albania','France','Italy','Montenegro','Slovakia','Cyprus','Russia','United Kingdom','Austria','Czech Republic',
 'Turkey','Romania','Finland','Hungary','Belarus','Denmark','Germany','Georgia','Slovenia']
 
 
 postlist = []
for country in countries:
    for submission in redd.subreddit("worldnews").search(country,limit=None):                                           
        post = {}
        post['title'] = str(submission.title)
        post["ratio"]= submission.upvote_ratio
        post["score"]= submission.score
        post["ups"] = round((submission.upvote_ratio*submission.score)/(2*submission.upvote_ratio - 1)) if submission.upvote_ratio != 0.5 else round(submission.score/2)
        post["downs"] = post["ups"] - submission.score
        post["comments"]= submission.num_comments
        post["id"]=submission.id
        post["country"]=country
        postlist.append(post)
        

Postdf = pd.DataFrame(postlist)


Postdf["Total interaction"]= Postdf["ups"]+Postdf["downs"]+Postdf["comments"]
Postdf["down ratio"]=Postdf["downs"]/(Postdf["ups"]+Postdf["downs"])

Postdf.rename(columns={"ratio":"up ratio"},inplace=True)
Postdf.dropna(inplace=True)



Q1 = Postdf.quantile(0.25)
Q3 = Postdf.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


downs= Postdf[["country","down ratio"]].groupby("country").agg(["count","mean"])

dist1= Postdf.groupby("country").mean()

sns.boxplot(x=dist1["comments"])



#from opinions-covid-part1
lowresp=['Austria','Germany', 'Ireland', 'Romania', 'Sweden', 'Switzerland']
highresp=['Georgia', 'Greece','Italy', 'Latvia', 'Ukraine', 'Macedonia']

dist1.reset_index(inplace=True)

import psycopg2
import pandas as pd

conn = psycopg2.connect(
    host='covid19db.org',
    port=5432,
    dbname='covid19',
    user='covid19',
    password='covid19')
cur = conn.cursor()


wave = "2008-2010"


questions=["E037"]

sql_command = """SELECT * FROM surveys WHERE wave=%(wave)s AND adm_area_1 IS NULL AND adm_area_2 IS NULL AND adm_area_3 IS NULL"""
df_surveys = pd.read_sql(sql_command, conn, params={'wave': wave})


import numpy as np
dataf= pd.DataFrame()
dataf["country"]=np.zeros(len(df_surveys))
dataf["countrycode"]=np.zeros(len(df_surveys))

for i in range(len(df_surveys)):
    dataf["country"].loc[i]=df_surveys["country"].loc[i]
    dataf["countrycode"].loc[i]=df_surveys["countrycode"].loc[i]
    for j in questions:
        title= df_surveys.properties[i][j]["Label"]
        liste=[]
        for k,v in df_surveys.properties[i][j]["Frequencies"].items():
            if k[-2]!="-":
                varie= int(k[-2:].replace("_",""))*v 
                liste.append(varie)
                dataf.at[i,title]=sum(liste)
                
                
 merged=pd.merge(dist1,dataf,how="left",on="country")
 
 for country in dist1["country"].unique():
    if country in lowresp:
        dist1.loc[dist1["country"]==country,"Responses"]= "low"
    else: 
        dist1.loc[dist1["country"]==country,"Responses"]= np.nan
        
 for country in dist1["country"].unique():
    if country in highresp:
        dist1.loc[dist1["country"]==country,"Responses"]= "high"


dist1.dropna(inplace=True)


dist1["comments"][dist1["Responses"]=="low"].mean()
dist1["comments"][dist1["Responses"]=="high"].mean()
              
sns.violinplot(y=dist1["comments"],x=dist1["Responses"]) 


merged.rename(columns={"Government responsibility": "responsibility"},inplace=True)
merged.rename(columns={"down ratio": "dratio"},inplace=True)
merged.rename(columns={"Total interaction": "tinteraction"},inplace=True)
merged["comm"]=merged["comments"]/merged["tinteraction"]

merged["ln_dratio"]=np.log(merged["dratio"])

plt.scatter(merged["responsibility"],merged["comm"])

form_1 = 'comm  ~ responsibility  '
fit_1 = smf.ols(formula = form_1, data = merged).fit()
fit_1.summary()

