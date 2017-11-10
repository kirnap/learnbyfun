# command line English Dictionary, written by Omer Kirnap
# Usage python eng_dict.py
import urllib2
import sys
import bs4
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
## I would like to thank Ozan Can Altiok for his suggestion
url2 = "http://googledictionary.freecollocation.com/meaning?word="


def get_mean(query):
    rep = urllib2.urlopen((url2+query)).read()
    soup = BeautifulSoup(rep, "html.parser")
    lst = soup.find_all('ol')
    #meaningful_text = soup.find('div', {'class' : 'std'}).text.split()
    for i in range(len(lst)-1):
	trimmed = [item for item in lst[i].text.split('\n') if len(item.strip())>0]
	ind = 0
	while ind < len(trimmed):
	    if trimmed[ind] == u'-':
		print u'   -%s' % trimmed[ind+1].strip()
		ind = ind+2
	    else:
		print trimmed[ind].strip()
		ind = ind+1

        




if __name__ == '__main__':
    get_mean(sys.argv[1])


	
# Dead code	
    # counter = 0
    # for item in meaningful_text:
    #     if item == '-' and counter == 0:
    #         print('\n  1.') , 
    #         counter += 2
    #     else:
    #         if counter == 0:
    #             print(item) ,
    #         else:
    #             if item == '-':
    #                 print('\n  '+str(counter)+'.') ,
    #             elif item[0].isupper() or item[0] == '(':
    #                 #print("Debug: ", item) ,
    #                 counter -= 1
    #                 print('\n' + item) ,
    #             else:
    #                 print(item) ,
