import urllib2
from bs4 import BeautifulSoup
import json
import pprint
import time
import csv

pp = pprint.PrettyPrinter()

# make server think that we are human
headers = {'User-Agent': 'Mozilla/5.0', }
tm2sp = {}
failed = list()


# value in gb pounds
def get_player_value(pid):
    url = urllib2.Request('http://www.transfermarkt.co.uk/players/marktwertverlauf/spieler/' + pid,
                          headers=headers)
    html = urllib2.urlopen(url).read()
    soup = BeautifulSoup(html, "lxml")
    js = soup.find_all("script")[-1].string.encode('utf-8')

    json_obj = None
    for line in js.split('\n'):
        if 'series' in line:
            json_obj = line

    player_values = {}
    if json_obj:
        json_result = '{' + json_obj.split(',', 1)[-1].split(',\'credits\'', 1)[0] + '}'
        json_result = json_result.replace('\'', '\"')
        json_data = json.loads(json_result)['series'][0]['data']


        for item in json_data:
            value = item['mw'].encode('utf-8', 'replace').replace('\xc2\xa3', '')
            if value != '-':
                value = int(value.replace('k', '000').replace('m', '0000').replace('.', ''))
            else:
                value = 0
            date = convert_date(str(item['datum_mw']))

            player_values[date] = value
    else:
        player_values['0'] = 0

    return player_values


def get_player_injuries(pid):
    url = urllib2.Request('http://www.transfermarkt.co.uk/player/verletzungen/spieler/' + pid,
                          headers=headers)
    html = urllib2.urlopen(url).read()
    soup = BeautifulSoup(html, "lxml")
    table = soup.find('table', attrs={'class': 'items'})
    player_injuries = {}
    if table:
        table_body = table.find('tbody')
        rows = table_body.find_all('tr')
        for row in rows:
            cols = row.find_all('td')
            injury = cols[1].text.strip().encode('utf-8')
            date_start = convert_date(cols[2].text.strip().encode('utf-8'))
            date_end = convert_date(cols[3].text.strip().encode('utf-8'))
            player_injuries[injury] = {'date_start': date_start, 'date_end': date_end}
    else:
        player_injuries['0'] = 0

    return player_injuries


def convert_date(tm_date):
    if tm_date == '-':
        date = time.gmtime()
        date = int(time.mktime(date))
    else:
        date = time.strptime(tm_date, "%b %d, %Y")
        date = int(time.mktime(date))
    return date


def parsed2csv(parsing_type):
    parsed = list()
    done = list()
    result_file = None
    if parsing_type == 'value':
        result_file = 'players_values.csv'
    elif parsing_type == 'injury':
        result_file = 'players_injuries.csv'
    with open("player_matches.csv") as players:
        count = 0
        is_header = True
        for row in players:
            if is_header:
                print "start"
                is_header = False
            else:
                player_id = row.split(";")[1]
                id_param = tm2sp[player_id]
                if id_param not in done:
                    try:
                        if parsing_type == 'value':
                            res = get_player_value(id_param)
                            for key, value in res.iteritems():
                                if int(key) != 0:
                                    parsed.append(str(player_id) + ";" + str(key) + ";" + str(value))
                        elif parsing_type == 'injury':
                            res = get_player_injuries(id_param)
                            for key, value in res.iteritems():
                                if str(key) != '0':
                                    parsed.append(str(player_id) + ";" + str(key) + ";" +
                                                  str(value['date_start']) + ";" + str(value['date_end']))
                        done.append(id_param)
                    except Exception, e:
                        print str(e)
                        failed.append(player_id)
            count += 1
            print "parsed " + str(len(done)) + " over " + str(count) + " rows"

    with open(result_file, "w") as players:
        count = 0
        for item in parsed:
            players.write("%s\n" % item)
            count += 1
            print "write: " + str(count) + " / " + str(len(parsed))


def main():
    with open("players_tm.csv") as players:
        reader = csv.DictReader(players, delimiter=';')
        for row in reader:
            tm2sp[row['id']] = row['tid']
    parsed2csv('injury')
    print "failed:"
    print failed


main()
