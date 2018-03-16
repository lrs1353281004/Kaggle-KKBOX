# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:35:42 2017
KKBOX  feature engineering
@author: Li Ruosong
"""
train=train.drop(['expiration_date'],axis=1)
test=test.drop(['expiration_date'],axis=1)
###too many nan values  in gender
train=train.drop(['gender'],axis=1)
test=test.drop(['gender'],axis=1)

print('adding single features')
###discrete song_year in different classes
def song_year_class(year):
    if type(year) == str :
        return round((int(year)-1918)/10)+1
    else:
        return 0
train['song_year_class']=train['song_year'].apply(song_year_class)
test['song_year_class']=test['song_year'].apply(song_year_class)
#train=train.drop(['song_year'],axis=1)
#test=test.drop(['song_year'],axis=1)
###remove age outliers 
train['bd']=train['bd'].replace(0,np.nan)  #此年龄属性中有很多值为0，但不应代表0岁，故暂时按缺失值处理
train['bd']=train['bd'].replace(range(100,219,1),np.nan)
test['bd']=test['bd'].replace(0,np.nan)
test['bd']=test['bd'].replace(range(100,219,1),np.nan)

###discrete age in different classes
def age_class(age):
    if (age>0 and age<100):
        return round(age/5)+1
    else:
        return 0
train['age_class']=train['bd'].apply(age_class)
test['age_class']=test['bd'].apply(age_class)
#train=train.drop(['bd'],axis=1)
#test=test.drop(['bd'],axis=1)
###discrete membership days in different classes
def member_days_class(days):
    if days<1:
        return 0
    elif days<10:
        return 1
    elif days<30:
        return 2
    elif days<180:
        return 3
    elif days<360:
        return 4
    elif days<1080:
        return 5
    elif days<1800:
        return 6
    else:
        return 7
train['member_days_class']=train['membership_days'].apply(member_days_class)
test['member_days_class']=test['membership_days'].apply(member_days_class)
#train=train.drop(['membership_days'],axis=1)
#test=test.drop(['membership_days'],axis=1)





###
def genre_id_count(x):
    if x == 'no_genre_id':
        return 0
    else:
        return x.count('|') + 1

train['genre_ids']=train['genre_ids'].replace(np.nan,'no_genre_id')
test['genre_ids']=test['genre_ids'].replace(np.nan,'no_genre_id')
train['genre_ids_count'] = train['genre_ids'].apply(genre_id_count).astype(np.int8)
test['genre_ids_count'] = test['genre_ids'].apply(genre_id_count).astype(np.int8)

###discrete genres_count in different classes
def genre_count_class(gen):
    if gen<1:
        return 0
    elif gen<2:
        return 1
    elif gen<5:
        return 2
    else:
        return 3
train['genre_count_class']=train['genre_ids_count'].apply(genre_count_class)
test['genre_count_class']=test['genre_ids_count'].apply(genre_count_class)
#train=train.drop(['genre_ids_count'],axis=1)
#test=test.drop(['genre_ids_count'],axis=1)







###
def lyricist_count(x):
    if x == 'no_lyricist':
        return 0
    else:
        s_count=1
        str_list=['|','/','\\',';','、','+','&' ]
        for str in str_list :
            s_count=s_count+x.count(str)
        return s_count
    

train['lyricist']=train['lyricist'].replace(np.nan,'no_lyricist')
test['lyricist']=test['lyricist'].replace(np.nan,'no_lyricist')
train['lyricists_count'] = train['lyricist'].apply(lyricist_count).astype(np.int8)
test['lyricists_count'] = test['lyricist'].apply(lyricist_count).astype(np.int8)

###discrete lyricist_count in different classes
def lyricist_count_class(cou):
    if cou<1:
        return 0
    elif cou<2:
        return 1
    elif cou<5:
        return 2
    elif cou<10:
        return 3
    else:
        return 4
train['lyricist_count_class']=train['lyricists_count'].apply(lyricist_count_class)
test['lyricist_count_class']=test['lyricists_count'].apply(lyricist_count_class)
#train=train.drop(['lyricists_count'],axis=1)
#test=test.drop(['lyricists_count'],axis=1)




###
def composer_count(x):
    if x == 'no_composer':
        return 0
    else:
        s_count=1
        str_list=['|','/','\\',';','、','+','&' ]
        for str in str_list :
            s_count=s_count+x.count(str)
        return s_count

train['composer']=train['composer'].replace(np.nan,'no_composer')
test['composer']=test['composer'].replace(np.nan,'no_composer')
train['composer_count'] = train['composer'].apply(composer_count).astype(np.int8)
test['composer_count'] = test['composer'].apply(composer_count).astype(np.int8)
###discrete composer_count in different classes
def composer_count_class(cou):
    if cou<1:
        return 0
    elif cou<2:
        return 1
    elif cou<5:
        return 2
    elif cou<10:
        return 3
    else:
        return 4
train['composer_count_class']=train['composer_count'].apply(composer_count_class)
test['composer_count_class']=test['composer_count'].apply(composer_count_class)
#train=train.drop(['composer_count'],axis=1)
#test=test.drop(['composer_count'],axis=1)



###
def artist_count(x):
    if x == 'no_artist':
        return 0
    else:
        return x.count('and') + x.count(',') + x.count('feat') + x.count('&')+1

train['artist_count'] = train['artist_name'].apply(artist_count).astype(np.int8)
test['artist_count'] = test['artist_name'].apply(artist_count).astype(np.int8)
###discrete artist_count in different classes
def artist_count_class(cou):
    if cou<1:
        return 0
    elif cou<2:
        return 1
    elif cou<5:
        return 2
    elif cou<10:
        return 3
    else:
        return 4
train['artist_count_class']=train['artist_count'].apply(artist_count_class)
test['artist_count_class']=test['artist_count'].apply(artist_count_class)
#train=train.drop(['artist_count'],axis=1)
#test=test.drop(['artist_count'],axis=1)

###
def song_length_class(x):
    if x < 20000:
        return 0
    elif x<600000:
        return round(x/120000)+1
    else:
        return 7

train['song_length_class'] = train['song_length'].apply(song_length_class).astype(np.int8)
test['song_length_class'] = test['song_length'].apply(song_length_class).astype(np.int8)

#train=train.drop(['song_length'],axis=1)
#test=test.drop(['song_length'],axis=1)


###summarize information in all samples(both train data  and  test data)
all_samples=pd.concat([train.drop(['target'],axis=1),test.drop(['id'],axis=1)])


###how often one user show up in all samples
user_fre_data=all_samples.groupby('msno').size()
def user_frequence(id):
    
    return user_fre_data[id]

train['user_frequence']=train['msno'].apply(user_frequence)
test['user_frequence']=test['msno'].apply(user_frequence)
del user_fre_data
### discrete user frequence into 6 classes
def user_frequence_class(fre):
    if fre<20:
        return 0
    elif fre<100:
        return 1
    elif fre<500:
        return 2
    elif fre<1000:
        return 3
    elif fre<=3000:
        return 4
    else:
        return 5

train['user_frequence_class']=train['user_frequence'].apply(user_frequence_class)
#train=train.drop(['user_frequence'],axis=1)
test['user_frequence_class']=test['user_frequence'].apply(user_frequence_class)
#test=test.drop(['user_frequence'],axis=1)

###how often one song  show up in all samples
song_fre_data=all_samples.groupby('song_id').size()
def song_frequence(id):
    return song_fre_data[id]

train['song_frequence']=train['song_id'].apply(song_frequence)
test['song_frequence']=test['song_id'].apply(song_frequence)
del song_fre_data
### discrete songs frequence into 7 classes
def song_frequence_class(fre):
    if fre<20:
        return 0
    elif fre<100:
        return 1
    elif fre<500:
        return 2
    elif fre<1000:
        return 3
    elif fre<=3000:
        return 4
    elif fre<=8000:
        return 5
    else:
        return 6

train['song_frequence_class']=train['song_frequence'].apply(song_frequence_class)
#train=train.drop(['song_frequence'],axis=1)
test['song_frequence_class']=test['song_frequence'].apply(song_frequence_class)
#test=test.drop(['song_frequence'],axis=1)


###how often the  singer  show up in all samples
singer_fre_data=all_samples.groupby('artist_name').size()
def artist_frequence(id):
    
    return singer_fre_data[id]
    

train['artist_frequence']=train['artist_name'].apply(artist_frequence)
test['artist_frequence']=test['artist_name'].apply(artist_frequence)

del singer_fre_data
### discrete singer frequence into 9 classes
def artist_frequence_class(fre):
    if fre<20:
        return 0
    elif fre<100:
        return 1
    elif fre<500:
        return 2
    elif fre<3000:
        return 3
    elif fre<=10000:
        return 4
    elif fre<=30000:
        return 5
    elif fre<=100000:
        return 6
    elif fre<=200000:
        return 7
    else:
        return 8
train['artist_frequence_class']=train['artist_frequence'].apply(artist_frequence_class)
#train=train.drop(['artist_frequence'],axis=1)
test['artist_frequence_class']=test['artist_frequence'].apply(artist_frequence_class)
#test=test.drop(['artist_frequence'],axis=1)


###how often the  composer  show up in all samples
composer_fre_data=all_samples.groupby('composer').size()
def composer_frequence(id):
    
    return composer_fre_data[id]
    

train['composer_frequence']=train['composer'].apply(composer_frequence)
test['composer_frequence']=test['composer'].apply(composer_frequence)

del composer_fre_data
### discrete composer frequence into 11 classes
def composer_frequence_class(fre):
    if fre<20:
        return 0
    elif fre<100:
        return 1
    elif fre<500:
        return 2
    elif fre<3000:
        return 3
    elif fre<=10000:
        return 4
    elif fre<=30000:
        return 5
    elif fre<=100000:
        return 6
    elif fre<=200000:
        return 7
    elif fre<=500000:
        return 8
    elif fre<=1000000:
        return 9
    else:
        return 10
train['composer_frequence_class']=train['composer_frequence'].apply(composer_frequence_class)
#train=train.drop(['composer_frequence'],axis=1)
test['composer_frequence_class']=test['composer_frequence'].apply(composer_frequence_class)
#test=test.drop(['composer_frequence'],axis=1)

###how often the  lyricist  show up in all samples
lyricist_fre_data=all_samples.groupby('lyricist').size()
def lyricist_frequence(id):
    
    return lyricist_fre_data[id]
    

train['lyricist_frequence']=train['lyricist'].apply(lyricist_frequence)
test['lyricist_frequence']=test['lyricist'].apply(lyricist_frequence)

del lyricist_fre_data
### discrete lyricist frequence into 11 classes
def lyricist_frequence_class(fre):
    if fre<20:
        return 0
    elif fre<100:
        return 1
    elif fre<500:
        return 2
    elif fre<3000:
        return 3
    elif fre<=10000:
        return 4
    elif fre<=30000:
        return 5
    elif fre<=100000:
        return 6
    elif fre<=200000:
        return 7
    elif fre<=500000:
        return 8
    elif fre<=1000000:
        return 9
    else:
        return 10
train['lyricist_frequence_class']=train['lyricist_frequence'].apply(lyricist_frequence_class)
#train=train.drop(['lyricist_frequence'],axis=1)
test['lyricist_frequence_class']=test['lyricist_frequence'].apply(lyricist_frequence_class)
#test=test.drop(['lyricist_frequence'],axis=1)
print('adding pair features')
###plus  pair  features
'''
mask=np.random.choice(7377418,1000000)
train=train.loc[mask]
test=test.loc[mask]
'''
train['user-song pair']=train['msno'].astype(str)+train['song_id'].astype(str)
test['user-song pair']=test['msno'].astype(str)+test['song_id'].astype(str)

train['user-genre pair']=train['msno'].astype(str)+train['genre_ids'].astype(str)
test['user-genre pair']=test['msno'].astype(str)+test['genre_ids'].astype(str)

train['user-artist pair']=train['msno'].astype(str)+train['artist_name'].astype(str)
test['user-artist pair']=test['msno'].astype(str)+test['artist_name'].astype(str)

train['user-composer pair']=train['msno'].astype(str)+train['composer'].astype(str)
test['user-composer pair']=test['msno'].astype(str)+test['composer'].astype(str)

train['user-lyricist pair']=train['msno'].astype(str)+train['lyricist'].astype(str)
test['user-lyricist pair']=test['msno'].astype(str)+test['lyricist'].astype(str)

train['user-language pair']=train['msno'].astype(str)+train['language'].astype(str)
test['user-language pair']=test['msno'].astype(str)+test['language'].astype(str)

train['user-source-system-tab pair']=train['msno'].astype(str)+train['source_system_tab'].astype(str)
test['user-source-system-tab pair']=test['msno'].astype(str)+test['source_system_tab'].astype(str)

train['user-source-screen-name pair']=train['msno'].astype(str)+train['source_screen_name'].astype(str)
test['user-source-screen-name pair']=test['msno'].astype(str)+test['source_screen_name'].astype(str)

train['user-source-type pair']=train['msno'].astype(str)+train['source_type'].astype(str)
test['user-source-type pair']=test['msno'].astype(str)+test['source_type'].astype(str)

##plus pair features  frequence analysis
all_samples=pd.concat([train.drop(['target'],axis=1),test.drop(['id'],axis=1)])
###
user_song_pair_fre_data=all_samples.groupby('user-song pair').size()
def user_song_pair_frequence(id):
    return user_song_pair_fre_data[id]

train['user_song_pair_frequence']=train['user-song pair'].apply(user_song_pair_frequence)
test['user_song_pair_frequence']=test['user-song pair'].apply(user_song_pair_frequence)
del user_song_pair_fre_data
###
user_genre_pair_fre_data=all_samples.groupby('user-genre pair').size()
def user_genre_pair_frequence(id):
    return user_genre_pair_fre_data[id]

train['user_genre_pair_frequence']=train['user-genre pair'].apply(user_genre_pair_frequence)
test['user_genre_pair_frequence']=test['user-genre pair'].apply(user_genre_pair_frequence)
del user_genre_pair_fre_data
###
user_artist_pair_fre_data=all_samples.groupby('user-artist pair').size()
def user_artist_pair_frequence(id):
    return user_artist_pair_fre_data[id]

train['user_artist_pair_frequence']=train['user-artist pair'].apply(user_artist_pair_frequence)
test['user_artist_pair_frequence']=test['user-artist pair'].apply(user_artist_pair_frequence)
del user_artist_pair_fre_data
###
user_composer_pair_fre_data=all_samples.groupby('user-composer pair').size()
def user_composer_pair_frequence(id):
    return user_composer_pair_fre_data[id]

train['user_composer_pair_frequence']=train['user-composer pair'].apply(user_composer_pair_frequence)
test['user_composer_pair_frequence']=test['user-composer pair'].apply(user_composer_pair_frequence)

del user_composer_pair_fre_data
###
user_lyricist_pair_fre_data=all_samples.groupby('user-lyricist pair').size()
def user_lyricist_pair_frequence(id):
    return user_lyricist_pair_fre_data[id]

train['user_lyricist_pair_frequence']=train['user-lyricist pair'].apply(user_lyricist_pair_frequence)
test['user_lyricist_pair_frequence']=test['user-lyricist pair'].apply(user_lyricist_pair_frequence)
del user_lyricist_pair_fre_data
###
user_language_pair_fre_data=all_samples.groupby('user-language pair').size()
def user_language_pair_frequence(id):
    return user_language_pair_fre_data[id]

train['user_language_pair_frequence']=train['user-language pair'].apply(user_language_pair_frequence)
test['user_language_pair_frequence']=test['user-language pair'].apply(user_language_pair_frequence)
del user_language_pair_fre_data
###
user_source_system_tab_fre_data=all_samples.groupby('user-source-system-tab pair').size()
def user_source_system_tab_fre_pair_frequence(id):
    return user_source_system_tab_fre_data[id]

train['user_source_system_tab_pair_frequence']=train['user-source-system-tab pair'].apply(user_source_system_tab_fre_pair_frequence)
test['user_source_system_tab_pair_frequence']=test['user-source-system-tab pair'].apply(user_source_system_tab_fre_pair_frequence)
del user_source_system_tab_fre_data
###
user_source_screen_name_fre_data=all_samples.groupby('user-source-screen-name pair').size()
def user_source_screen_name_fre_pair_frequence(id):
    return user_source_screen_name_fre_data[id]

train['user_source_screen_name_pair_frequence']=train['user-source-screen-name pair'].apply(user_source_screen_name_fre_pair_frequence)
test['user_source_screen_name_pair_frequence']=test['user-source-screen-name pair'].apply(user_source_screen_name_fre_pair_frequence)
del user_source_screen_name_fre_data
###

user_source_type_fre_data=all_samples.groupby('user-source-type pair').size()
def user_source_type_fre_pair_frequence(id):
    return user_source_type_fre_data[id]

train['user_source_type_pair_frequence']=train['user-source-type pair'].apply(user_source_type_fre_pair_frequence)
test['user_source_type_pair_frequence']=test['user-source-type pair'].apply(user_source_type_fre_pair_frequence)
del user_source_type_fre_data
#######   pair_frequence_class
###
def user_genre_pair_class(fre):
    if fre<2:
        return 0
    elif fre<5:
        return 1
    elif fre<20:
        return 2
    elif fre<50:
        return 3
    elif fre<100:
        return 4
    elif fre<200:
        return 5
    else:
        return 6
train['user_genre_pair_fre_class']=train['user_genre_pair_frequence'].apply(user_genre_pair_class)
test['user_genre_pair_fre_class']=test['user_genre_pair_frequence'].apply(user_genre_pair_class)
###
def user_artist_pair_class(fre):
    if fre<2:
        return 0
    elif fre<5:
        return 1
    elif fre<10:
        return 2
    elif fre<20:
        return 3
    elif fre<50:
        return 4
    else:
        return 5
train['user_artist_pair_fre_class']=train['user_artist_pair_frequence'].apply(user_artist_pair_class)
test['user_artist_pair_fre_class']=test['user_artist_pair_frequence'].apply(user_artist_pair_class)
###
def user_composer_pair_class(fre):
    if fre<2:
        return 0
    elif fre<10:
        return 1
    elif fre<50:
        return 2
    elif fre<100:
        return 3
    elif fre<200:
        return 4
    else:
        return 5
train['user_composer_pair_fre_class']=train['user_composer_pair_frequence'].apply(user_composer_pair_class)
test['user_composer_pair_fre_class']=test['user_composer_pair_frequence'].apply(user_composer_pair_class)
###
def user_lyricist_pair_class(fre):
    if fre<2:
        return 0
    elif fre<10:
        return 1
    elif fre<50:
        return 2
    elif fre<100:
        return 3
    elif fre<200:
        return 4
    elif fre<500:
        return 5
    else:
        return 6
train['user_lyricist_pair_fre_class']=train['user_lyricist_pair_frequence'].apply(user_lyricist_pair_class)
test['user_lyricist_pair_fre_class']=test['user_lyricist_pair_frequence'].apply(user_lyricist_pair_class)
###
def user_language_pair_class(fre):
    if fre<2:
        return 0
    elif fre<10:
        return 1
    elif fre<50:
        return 2
    elif fre<100:
        return 3
    elif fre<200:
        return 4
    elif fre<500:
        return 5
    else:
        return 6
train['user_language_pair_fre_class']=train['user_language_pair_frequence'].apply(user_language_pair_class)
test['user_language_pair_fre_class']=test['user_language_pair_frequence'].apply(user_language_pair_class)
###
def user_source_system_tab_pair_class(fre):
    if fre<2:
        return 0
    elif fre<10:
        return 1
    elif fre<50:
        return 2
    elif fre<100:
        return 3
    elif fre<200:
        return 4
    elif fre<500:
        return 5
    else:
        return 6
train['user_source_system_tab_pair_fre_class']=train['user_source_system_tab_pair_frequence'].apply(user_source_system_tab_pair_class)
test['user_source_system_tab_pair_fre_class']=test['user_source_system_tab_pair_frequence'].apply(user_source_system_tab_pair_class)

###
def user_source_screen_name_pair_class(fre):
    if fre<2:
        return 0
    elif fre<10:
        return 1
    elif fre<50:
        return 2
    elif fre<100:
        return 3
    elif fre<200:
        return 4
    elif fre<500:
        return 5
    else:
        return 6
train['user_source_screen_name_pair_fre_class']=train['user_source_screen_name_pair_frequence'].apply(user_source_screen_name_pair_class)
test['user_source_screen_name_pair_fre_class']=test['user_source_screen_name_pair_frequence'].apply(user_source_screen_name_pair_class)
###
def user_source_type_pair_class(fre):
    if fre<2:
        return 0
    elif fre<10:
        return 1
    elif fre<50:
        return 2
    elif fre<100:
        return 3
    elif fre<200:
        return 4
    elif fre<500:
        return 5
    else:
        return 6
train['user_source_type_pair_fre_class']=train['user_source_type_pair_frequence'].apply(user_source_type_pair_class)
test['user_source_type_pair_fre_class']=test['user_source_type_pair_frequence'].apply(user_source_type_pair_class)






train=train.drop(['user-song pair'],axis=1)
test=test.drop(['user-song pair'],axis=1)
train=train.drop(['user-genre pair'],axis=1)
test=test.drop(['user-genre pair'],axis=1)
train=train.drop(['user-artist pair'],axis=1)
test=test.drop(['user-artist pair'],axis=1)
train=train.drop(['user-composer pair'],axis=1)
test=test.drop(['user-composer pair'],axis=1)
train=train.drop(['user-lyricist pair'],axis=1)
test=test.drop(['user-lyricist pair'],axis=1)
train=train.drop(['user-language pair'],axis=1)
test=test.drop(['user-language pair'],axis=1)
train=train.drop(['user-source-system-tab pair'],axis=1)
test=test.drop(['user-source-system-tab pair'],axis=1)
train=train.drop(['user-source-screen-name pair'],axis=1)
test=test.drop(['user-source-screen-name pair'],axis=1)
train=train.drop(['user-source-type pair'],axis=1)
test=test.drop(['user-source-type pair'],axis=1)

del all_samples
#############################################################################
###adding ratio features
train['user_song_ratio']=train['user_song_pair_frequence']/train['user_frequence']
test['user_song_ratio']=test['user_song_pair_frequence']/test['user_frequence']

train['user_genre_ratio']=train['user_genre_pair_frequence']/train['user_frequence']
test['user_genre_ratio']=test['user_genre_pair_frequence']/test['user_frequence']

train['user_artist_ratio']=train['user_artist_pair_frequence']/train['user_frequence']
test['user_artist_ratio']=test['user_artist_pair_frequence']/test['user_frequence']

train['user_composer_ratio']=train['user_composer_pair_frequence']/train['user_frequence']
test['user_composer_ratio']=test['user_composer_pair_frequence']/test['user_frequence']

train['user_lyricist_ratio']=train['user_lyricist_pair_frequence']/train['user_frequence']
test['user_lyricist_ratio']=test['user_lyricist_pair_frequence']/test['user_frequence']

train['user_language_ratio']=train['user_language_pair_frequence']/train['user_frequence']
test['user_language_ratio']=test['user_language_pair_frequence']/test['user_frequence']

train['user_source_system_tab_ratio']=train['user_source_system_tab_pair_frequence']/train['user_frequence']
test['user_source_system_tab_ratio']=test['user_source_system_tab_pair_frequence']/test['user_frequence']

train['user_source_screen_name_ratio']=train['user_source_screen_name_pair_frequence']/train['user_frequence']
test['user_source_screen_name_ratio']=test['user_source_screen_name_pair_frequence']/test['user_frequence']

train['user_source_type_ratio']=train['user_source_type_pair_frequence']/train['user_frequence']
test['user_source_type_ratio']=test['user_source_type_pair_frequence']/test['user_frequence']

def ratio_class(ratio):
    if ratio<0.0002:
        return 0
    elif ratio<0.0005:
        return 1
    elif ratio<0.001:
        return 2
    elif ratio<0.005:
        return 3
    elif ratio<0.01:
        return 4
    elif ratio<0.05:
        return 5
    elif ratio<0.1:
        return 6  
    elif ratio<0.2:
        return 7
    elif ratio<0.5:
        return 8
    elif ratio<0.8:
        return 9
    else:
        return 10
train['user_song_ratio_class']=train['user_song_ratio'].apply(ratio_class)
test['user_song_ratio_class']=test['user_song_ratio'].apply(ratio_class)

train['user_genre_ratio_class']=train['user_genre_ratio'].apply(ratio_class)
test['user_genre_ratio_class']=test['user_genre_ratio'].apply(ratio_class)

train['user_artist_ratio_class']=train['user_artist_ratio'].apply(ratio_class)
test['user_artist_ratio_class']=test['user_artist_ratio'].apply(ratio_class)

train['user_composer_ratio_class']=train['user_composer_ratio'].apply(ratio_class)
test['user_composer_ratio_class']=test['user_composer_ratio'].apply(ratio_class)

train['user_lyricist_ratio_class']=train['user_lyricist_ratio'].apply(ratio_class)
test['user_lyricist_ratio_class']=test['user_lyricist_ratio'].apply(ratio_class)

train['user_language_ratio_class']=train['user_language_ratio'].apply(ratio_class)
test['user_language_ratio_class']=test['user_language_ratio'].apply(ratio_class)

train['user_source_system_tab_ratio_class']=train['user_source_system_tab_ratio'].apply(ratio_class)
test['user_source_system_tab_ratio_class']=test['user_source_system_tab_ratio'].apply(ratio_class)

train['user_source_screen_name_ratio_class']=train['user_source_screen_name_ratio'].apply(ratio_class)
test['user_source_screen_name_ratio_class']=test['user_source_screen_name_ratio'].apply(ratio_class)

train['user_source_type_ratio_class']=train['user_source_type_ratio'].apply(ratio_class)
test['user_source_type_ratio_class']=test['user_source_type_ratio'].apply(ratio_class)
#####ratio sum
train['ratio_sum_song']=train['user_song_ratio']+train['user_genre_ratio']+train['user_artist_ratio']+train['user_composer_ratio']+train['user_lyricist_ratio']
test['ratio_sum_song']=test['user_song_ratio']+test['user_genre_ratio']+test['user_artist_ratio']+test['user_composer_ratio']+test['user_lyricist_ratio']


train['fre_sum_song']=train['user_song_pair_frequence']+train['user_genre_pair_frequence']+train['user_artist_pair_frequence']+train['user_composer_pair_frequence']+train['user_lyricist_pair_frequence']
test['fre_sum_song']=test['user_song_pair_frequence']+test['user_genre_pair_frequence']+test['user_artist_pair_frequence']+test['user_composer_pair_frequence']+test['user_lyricist_pair_frequence']

def ratio_sum_class(ratio):
    if ratio<0.001:
        return 0
    elif ratio<0.003:
        return 1
    elif ratio<0.008:
        return 2
    elif ratio<0.015:
        return 3
    elif ratio<0.03:
        return 4
    elif ratio<0.09:
        return 5
    elif ratio<0.3:
        return 6  
    elif ratio<0.7:
        return 7
    elif ratio<1.5:
        return 8
    elif ratio<3:
        return 9
    elif ratio<4.9:
        return 10
    else:
        return 11

train['ratio_sum_song_class']=train['ratio_sum_song'].apply(ratio_sum_class)
test['ratio_sum_song_class']=test['ratio_sum_song'].apply(ratio_sum_class)

print ("Done adding features")
'''
path='E:/workspace/Kaggle/KKBOX_recommend/code_files/individual_parts'
train.to_csv(path + 'train_FE.csv', index=True)
test.to_csv(path + 'test_FE.csv',  index=True)


'''

