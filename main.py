#!/usr/local/bin/python3.5
import urllib.request as urllib2
import codecs
from bs4 import BeautifulSoup
import json
import pprint
import time
import csv

pp = pprint.PrettyPrinter()


def convert_date(tm_date):
    if tm_date == '-':
        date = time.gmtime()
        date = int(time.mktime(date))
    else:
        date = time.strptime(tm_date, "%b %d, %Y")
        date = int(time.mktime(date))
    return date


def convert_value(tm_value):
    tm_value = tm_value.decode("utf-8")
    if tm_value != '-':
        value = tm_value.replace('£', '').replace('Â', '')
        value = int(value.replace('k', '000').replace('m', '0000').replace('.', ''))
    else:
        value = 0
    return value


class TransfermarktParser:
    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0', }
        self.tm2sp = {}
        self.failed = list()

    def get_player_value(self, pid):
        url = urllib2.Request('http://www.transfermarkt.co.uk/players/marktwertverlauf/spieler/{pid}'.format(
           pid=pid), headers=self.headers)
        html = urllib2.urlopen(url).read()
        soup = BeautifulSoup(html, "lxml")
        js = soup.find_all("script")[-1].string
        for line in js.split('\n'):
            if 'series' in line:
                json_obj = line
        player_values = {}
        if json_obj:
            json_result = '{' + json_obj.split(',', 1)[-1].split(',\'credits\'', 1)[0] + '}'
            json_result = json_result.replace('\'', '\"')
            json_data = json.loads(json_result)['series'][0]['data']

            for item in json_data:
                value = item['mw'].encode('utf-8', 'replace')
                date = convert_date(str(item['datum_mw']))
                player_values[date] = convert_value(value)
        else:
            player_values['0'] = 0

        return player_values

    def get_player_injuries(self, pid):
        url = urllib2.Request('http://www.transfermarkt.co.uk/player/verletzungen/spieler/{pid}'.format(
           pid=pid), headers=self.headers)
        html = urllib2.urlopen(url).read()
        soup = BeautifulSoup(html, "lxml")
        table = soup.find('table', attrs={'class': 'items'})
        player_injuries = {}
        if table:
            table_body = table.find('tbody')
            rows = table_body.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                injury = cols[1].text.strip()
                date_start = convert_date(cols[2].text.strip())
                date_end = convert_date(cols[3].text.strip())
                player_injuries[injury] = {'date_start': date_start, 'date_end': date_end}
        else:
            player_injuries['0'] = 0

        return player_injuries

    def get_mean_value(self, tourn_id):
        url = urllib2.Request('http://www.transfermarkt.co.uk/tournament/startseite/wettbewerb/{tourn_id}'.format(
           tourn_id=tourn_id), headers=self.headers)
        html = urllib2.urlopen(url).read()
        soup = BeautifulSoup(html, "lxml")
        table = soup.find('table', attrs={'class': 'profilheader'})
        if table:
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                mean_value = cols[0].text.strip().encode('utf-8')
        return convert_value(mean_value)

    def parsed2csv(self, parsing_type):
        with open("players_tm.csv") as players:
            reader = csv.DictReader(players, delimiter=';')
            for row in reader:
                self.tm2sp[row['id']] = row['tid']
        parsed = list()
        done = list()
        result_file = None
        if parsing_type == 'value':
            result_file = 'players_values2.csv'
        elif parsing_type == 'injury':
            result_file = 'players_injuries.csv'
        with open("players_matches.csv") as players:
            count = 0
            is_header = True
            for row in players:
                if is_header:
                    print("start")
                    is_header = False
                else:
                    player_id = row.split(";")[1]
                    id_param = self.tm2sp[player_id]
                    if id_param not in done:
                        try:
                            if parsing_type == 'value':
                                res = self.get_player_value(id_param)
                                for key, value in res.items():
                                    if int(key) != 0:
                                        parsed.append("{player_id};{key};{value}".format(
                                            player_id=str(player_id), key=str(key), value=str(value)))
                            elif parsing_type == 'injury':
                                res = self.get_player_injuries(id_param)
                                for key, value in res.items():
                                    if str(key) != '0':
                                        parsed.append("{player_id};{key};{date_start};{date_end}".format(
                                            player_id=str(player_id), key=str(key),
                                            date_start=str(value['date_start']), date_end=str(value['date_end'])
                                        ))
                            done.append(id_param)
                        except Exception as e:
                            print(str(e))
                            self.failed.append(player_id)
                count += 1
                print("parsed {player_count} player over {row_count} rows".format(
                    player_count=str(len(done)), row_count=str(count)))

        with open(result_file, "w") as players:
            count = 0
            parsed_count = len(parsed)
            for item in parsed:
                players.write("%s\n" % item)
                count += 1
                print("write: {count}/{of}".format(count=str(count), of=str(parsed_count)))

    def update_csv(self):
        with codecs.open("tournaments.csv", "r", encoding='utf-8', errors='ignore') as tournaments:
            is_header = True
            parsed = list()
            for row in tournaments:
                try:
                    if is_header:
                        print("start")
                        is_header = False
                        tourn_sp_id = "id"
                        value = "tourn_mean"
                    else:
                        tourn_sp_id = row.strip("\n").split(";")[0]
                        tourn_tm_id = row.strip("\n").split(";")[2]
                        value = self.get_mean_value(tourn_tm_id)
                    parsed.append("{id};{tourn_mean}".format(id=tourn_sp_id, tourn_mean=value))
                except Exception as e:
                    print(str(e))
                    print(row)
        with open("tourn_with_values.csv", "w") as players:
            for item in parsed:
                players.write("%s\n" % item)


parser = TransfermarktParser()
parser.parsed2csv('injury')

