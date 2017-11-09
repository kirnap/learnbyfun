# command line English Dictionary, written by Omer Kirnap
# Usage python eng_dict.py
import urllib2
import sys
from bs4 import BeautifulSoup
#import urllib

#url = "https://www.google.com/search?"

#query = raw_input("What do you want to look for meaning? >> ")

#query = urllib.urlencode( {'q' : query } )

# Here urlopen returns 403 forbidden 
#response = urllib2.urlopen (url + query ).read()

# I couldn't be able to succesfully retrieve the text that I want,
# I checked with one of the online html editor to retrieve html document back and couldn't see the same output page, that is not same output with google-chrome
#response2 = requests.get(url + query)
#soup = BeautifulSoup(response2.content)


##### Solution from another page
url2 = "http://googledictionary.freecollocation.com/meaning?word="
#query2 = raw_input("What do you want to look for meaning? >> ")


def get_mean(query):
    rep = urllib2.urlopen((url2+query)).read()
    soup = BeautifulSoup(rep)
    meaningful_text = soup.find('div', {'class' : 'std'}).text.split()
    counter = 0
    for item in meaningful_text:
        if item == '-' and counter == 0:
            print('\n  1.') , 
            counter += 2
        else:
            if counter == 0:
                print(item) ,
            else:
                if item == '-':
                    print('\n  '+str(counter)+'.') ,
                elif item[0].isupper():
                    counter -= 1
                    print('\n' + item) ,
                else:
                    print(item) ,


if __name__ == '__main__':
    get_mean(sys.argv[1])

