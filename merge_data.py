# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 13:48:52 2017
KKBOX merge data 
@author: Li Ruosong
"""
train = train.merge(songs, on='song_id', how='left')
test = test.merge(songs, on='song_id', how='left')
del songs
members['membership_days'] = members['expiration_date'].subtract(members['registration_init_time']).dt.days.astype(int)

members['registration_year'] = members['registration_init_time'].dt.year
members['registration_month'] = members['registration_init_time'].dt.month
members['registration_day'] = members['registration_init_time'].dt.day

members['expiration_year'] = members['expiration_date'].dt.year
members['expiration_month'] = members['expiration_date'].dt.month
members['expiration_day'] = members['expiration_date'].dt.day
members = members.drop(['registration_init_time'], axis=1)

train = train.merge(members, on='msno', how='left')
test = test.merge(members, on='msno', how='left')
del members
#dig out year and country information in isrc of a song
def isrc_to_year(isrc):
    if type(isrc) == str:
        if int(isrc[5:7]) > 17:
            return str(1900 + int(isrc[5:7]))
        else:
            return str(2000 + int(isrc[5:7]))
    else:
        return np.nan
        
def isrc_to_country(isrc):
    if type(isrc) == str:
        
        return isrc[0:2]
    else:
        return np.nan
        
songs_extra['song_year'] = songs_extra['isrc'].apply(isrc_to_year)
songs_extra['song_country'] = songs_extra['isrc'].apply(isrc_to_country)
songs_extra.drop(['isrc', 'name'], axis = 1, inplace = True)



train = train.merge(songs_extra, on = 'song_id', how = 'left') #Only keep all the data in the train table
train.song_length.fillna(200000,inplace=True)
train.song_length = train.song_length.astype(np.uint32)
train.song_id = train.song_id.astype('category')


test = test.merge(songs_extra, on = 'song_id', how = 'left')
test.song_length.fillna(200000,inplace=True)
test.song_length = test.song_length.astype(np.uint32)
test.song_id = test.song_id.astype('category')

del songs_extra