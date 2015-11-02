import csv

tm2sp = list()
with open("players_tm.csv") as players:
    reader = csv.DictReader(players, delimiter=';')
    for row in reader:
        tm2sp.append(row['id'])

result = list()
result.append(
    '"id";"player_id";"team_id";"amplua";"match_id";"minute_started";"minute_finished";"goals";"goal_passes";"yellow_cards";"red_cards";"yellow_cards_on_him";"red_cards_on_him";"fols";"fols_on_him";"obvodki";"otbor_all";"earned_penalty";"strikes";"precise_strikes";"framework_strikes";"penalty";"offside";"autogol";"conceded_goals_with_him";"out_of_gate_win";"saves";"gk_penalty_saves";"whitewash_match"')

with open("players.csv") as players:
    count = 0
    for row in players:
        if row.split(";")[1] in tm2sp:
            result.append(row)
        count += 1
        print "read: " + str(count)


with open('only_linked_players.csv', "w") as players:
    count = 0
    for item in result:
        players.write("%s\n" % item)
        count += 1
        print "write: " + str(count)
