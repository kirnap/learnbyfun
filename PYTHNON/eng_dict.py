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

def retrieve_sentence(ugly_text):
    """
    Retrieves a sentence from a given list
    """
    alist = ugly_text.text.split()
    sents = []
    sent = ""
    for item in alist:
        if item != "-" and item != alist[-1]:
            sent += item + " "
        #elif item == alist[-1]:
        #    sents.append(sent)
        else:
            sents.append(sent)
            sent = ""
    return sents



def get_mean(query):
    rep = urllib2.urlopen((url2+query)).read()
    soup = BeautifulSoup(rep, "html.parser")
    #meaningful_text = soup.find('div', {'class' : 'std'}).text.split()
    #color:#767676;list-style:none
    all_examples = soup.find_all(attrs={"style" : "color:#767676;list-style:none"})
    alltxt = soup.find_all(attrs={"style" : "list-style:decimal"})
    for i in range(len(alltxt) - 1):
        s = list(retrieve_sentence(alltxt[i]) - retrieve_sentence(alltxt[i+1]))
        for item in s:
            print item
        
        

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


if __name__ == '__main__':
    get_mean(sys.argv[1])

