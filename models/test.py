import os
import pandas
import urllib

def download_little_file(from_url,to_path):
    conn = urllib.request.urlopen(from_url).read()
    f = open(to_path,'wb')
    f.write(conn.read())
    f.close()

def test_dl():
    url = 'https://images.sex.com/images/pinporn/2012/06/15/620/326769.jpg'
    path = os.path.join('./',url)
    download_little_file(url,path)
    