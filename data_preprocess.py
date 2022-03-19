import json
import math
import re
import pandas as pd

team_name_to_location = {}
team_name_to_location['atlanta hawks'] = ('State Farm Arena', 'Atlanta, Georgia')
team_name_to_location['boston celtics'] = ('TD Garden', 'Boston, Massachusetts')
team_name_to_location['brooklyn nets'] = ('Barclays Center', 'Brooklyn, New York')
team_name_to_location['charlotte hornets'] = ('Spectrum Center', 'Charlotte, North Carolina')
team_name_to_location['chicago bulls'] = ('United Center', 'Chicago, Illinois')
team_name_to_location['cleveland cavaliers'] = ('Rocket Mortgage Fieldhouse', 'Cleveland, Ohio')
team_name_to_location['dallas mavericks'] = ('American Airlines Center', 'Dallas, Texas')
team_name_to_location['denver nuggets'] = ('Pepsi Center', 'Denver, Colorado')
team_name_to_location['detroit pistons'] = ('Little Caesars Arena', 'Detroit, Michigan')
team_name_to_location['golden_state warriors'] = ('Chase Center', 'San Francisco, California')
team_name_to_location['houston rockets'] = ('Toyota Center', 'Houston, Texas')
team_name_to_location['indiana pacers'] = ('Bankers Life Fieldhouse', 'Indianapolis, Indiana')
team_name_to_location['los_angeles clippers'] = ('Staples Center', 'Los Angeles, California')
team_name_to_location['los_angeles lakers'] = ('Staples Center', 'Los Angeles, California')
team_name_to_location['memphis grizzlies'] = ('Fedex Forum', 'Memphis, Tennessee')
team_name_to_location['miami heat'] = ('American Airlines Arena', 'Miami, Florida')
team_name_to_location['milwaukee bucks'] = ('Fiserv Forum', 'Milwaukee, Wisconsin')
team_name_to_location['minnesota timberwolves'] = ('Target Center', 'Minneapolis, Minnesota')
team_name_to_location['new_orleans pelicans'] = ('Smoothie King Center', 'New Orleans, Louisiana')
team_name_to_location['new_york knicks'] = ('Madison Square Garden', 'New York')
team_name_to_location['oklahoma_city thunder'] = ('Chesapeake Energy Arena', 'Oklahoma')
team_name_to_location['orlando magic'] = ('Amway Center', 'Orlando, Florida')
team_name_to_location['philadelphia 76ers'] = ('Wells Fargo Center', 'Philadelphia, Pennsylvania')
team_name_to_location['phoenix suns'] = ('Talking Stick Resort Arena', 'Phoenix, Arizona')
team_name_to_location['portland trail_blazers'] = ('Moda Center', 'Portland, Oregon')
team_name_to_location['sacramento kings'] = ('Golden 1 Center', 'Sacramento, California')
team_name_to_location['san_antonio spurs'] = ('AT&T Center', 'San Antonio, Texas')
team_name_to_location['toronto raptors'] = ('Scotiabank Arena', 'Toronto, Ontario')
team_name_to_location['utah jazz'] = ('Vivint Smart Home Arena', 'Salt Lake City, Utah')
team_name_to_location['washington wizards'] = ('Capital One Arena', 'Washington, D.C.')


def read_position_csv(file_name):
    return pd.read_csv(file_name)


players_position_data = read_position_csv('./inputs/player_to_position.csv')
players_position_data = dict(zip(players_position_data['PLAYER'].str.lower(), players_position_data['POS']))


class Player:
    def __init__(self, player_stats_dict):
        self.player_stats_dict = player_stats_dict

    def add_stat(self, stat_name, value):
        self.player_stats_dict[stat_name] = value

    def is_home(self):
        return self.player_stats_dict['IS_HOME'] == 'yes'


class Team:
    def __init__(self, team_stats_dict, players):
        self.team_stats_dict = team_stats_dict
        self.players = players
        self.filter_players()

    def filter_players(self):
        self.players.sort(key=lambda player: int(player.player_stats_dict['MIN']), reverse=True)
        players = [player for player in self.players if player.player_stats_dict['STARTER'] == 'yes']
        other_players = [p for p in self.players if p.player_stats_dict['STARTER'] == 'no'][:2]
        self.players = players + other_players

    def add_stat(self, stat_name, value):
        self.team_stats_dict[stat_name] = value

    def get_team_arena_and_location(self):
        self.arena_and_location = team_name_to_location[
            self.team_stats_dict['TEAM_PLACE'].lower() + ' ' + self.team_stats_dict['TEAM_NAME'].lower()]


def process_player(entry_data):
    player = Player({})
    for stat in entry_data:
        value, stat_name = stat.split('￨')
        player.add_stat(stat_name, value)
    return player


def process_team(entry_data, players_list):
    team = Team({}, players_list)
    for stat in entry_data[:-1]:
        value, stat_name = stat.split('￨')
        team.add_stat(stat_name, value)
    team.get_team_arena_and_location()
    return team


def process_game_stats(game_data):
    home_team_players = []
    away_team_players = []
    game_data = game_data.split('<ent>￨<ent>')[1:]
    for index, entry in enumerate(game_data):
        if index == len(game_data) - 1:
            entry_data = entry.split(' ')[1:]
        else:
            entry_data = entry.split(' ')[1:-1]
        if '<blank>￨<blank>' not in entry_data[-1]:
            player = process_player(entry_data)
            if player.is_home():
                home_team_players.append(player)
            else:
                away_team_players.append(player)
        else:
            if entry_data[-2] == 'yes￨IS_HOME':
                home_team = process_team(entry_data, home_team_players)
            else:
                away_team = process_team(entry_data, away_team_players)
    return home_team, away_team


def get_team_info(team_dict):
    city = team_dict["TEAM_PLACE"]
    name = team_dict["TEAM_NAME"]
    wins = int(team_dict["TEAM_WINS"])
    losses = int(team_dict["TEAM_LOSSES"])
    pts = int(team_dict["TEAM-PTS"])
    pts_q1 = int(team_dict["TEAM-PTS-QTR1"])
    pts_q2 = int(team_dict["TEAM-PTS-QTR2"])
    pts_q3 = int(team_dict["TEAM-PTS-QTR3"])
    pts_q4 = int(team_dict["TEAM-PTS-QTR4"])
    fgm = int(team_dict["TEAM-FGM"])
    fga = int(team_dict["TEAM-FGA"])
    fg_pct = math.floor(float(team_dict["TEAM-FG_PCT"]) * 100)
    fg3m = int(team_dict["TEAM-FG3M"])
    fg3a = int(team_dict["TEAM-FG3A"])
    fg3_pct = math.floor(float(team_dict["TEAM-FG3_PCT"]) * 100)
    ftm = int(team_dict["TEAM-FTM"])
    fta = int(team_dict["TEAM-FTA"])
    ft_pct = math.floor(float(team_dict["TEAM-FT_PCT"]) * 100)
    reb = int(team_dict["TEAM-REB"])
    ast = int(team_dict["TEAM-AST"])
    tov = int(team_dict["TEAM-TOV"])

    return city, name, wins, losses, pts, pts_q1, pts_q2, pts_q3, pts_q4, fgm, fga, fg_pct, fg3m, fg3a, fg3_pct, ftm, fta, ft_pct, reb, ast, tov


def get_quarter_score_line(quarter, home_full_name, vis_full_name, home_q_pts, vis_q_pts, paragraph):
    quarter_score_win = " the %s outscored the %s %d - %d in the %s quarter "
    quarter_score_tie = " the game was tied at %d at the end of the %s quarter "
    home_won = home_q_pts > vis_q_pts
    tie = home_q_pts == vis_q_pts
    if home_won:
        paragraph.append(quarter_score_win % (home_full_name, vis_full_name, home_q_pts, vis_q_pts, quarter))
    elif tie:
        paragraph.append(quarter_score_tie % (home_q_pts, quarter))
    else:
        paragraph.append(quarter_score_win % (vis_full_name, home_full_name, vis_q_pts, home_q_pts, quarter))


def get_player_position(player_name):
    player_name = player_name.replace("_", " ")
    player_name = player_name.lower()

    if player_name not in players_position_data:
        return ""
    return players_position_data[player_name]


def get_players_line(players, team_name, paragraph):
    player_line = "%s , playing for %s , %s , played %d minutes , scored %d points ( %d - %d with %d percent from " \
                  "the field , %d - %d with %d percent from three point range and %d - %d with %d percent from free " \
                  "throw line ) , he contributed %d rebounds ( %d offensive rebounds ) , %d assists, %d steals , " \
                  "%d blocks , and committed %d fouls and %d turnovers "
    for player in players:
        player_data = player.player_stats_dict
        player_full_name = player_data['FIRST_NAME'] + " " + player_data['LAST_NAME']
        position = get_player_position(player_full_name)
        if position != '':
            if player_data['STARTER'] == "yes":
                position_passage = "started at " + position
            else:
                position_passage = "came off the bench as " + position
        else:
            if player_data['STARTER'] == "yes":
                position_passage = "as a starter"
            else:
                position_passage = "off the bench"

        player_line_values = player_line % (player_full_name, team_name, position_passage,
                                            int(player_data["MIN"]), int(player_data["PTS"]),
                                            int(player_data["FGM"]), int(player_data["FGA"]),
                                            math.floor(float(
                                                0 if player_data['FG_PCT'] == 'N/A' else player_data['FG_PCT']) * 100),
                                            int(player_data["FG3M"]), int(player_data["FG3A"]),
                                            math.floor(float(
                                                0 if player_data['FG3_PCT'] == 'N/A' else player_data[
                                                    'FG3_PCT']) * 100),
                                            int(player_data["FTM"]), int(player_data["FTA"]),
                                            math.floor(float(
                                                0 if player_data['FT_PCT'] == 'N/A' else player_data['FT_PCT']) * 100),
                                            int(player_data["REB"]), int(player_data["OREB"]),
                                            int(player_data["AST"]), int(player_data["STL"]),
                                            int(player_data["BLK"]), int(player_data["PF"]), int(player_data['TO']))

        player_line_values = player_line_values.replace("0 - 0 shooting with 0 percent , ", "")
        player_line_values = player_line_values.replace("0 - 0 with 0 percent from three point range and ", "")
        player_line_values = player_line_values.replace("0 - 0 with 0 percent from free throw line", "")

        player_line_values = player_line_values.replace("()", "")
        player_line_values = player_line_values.replace("( )", "")
        player_line_values = player_line_values.replace("(  )", "")
        player_line_values = player_line_values.replace("  ", "")
        player_line_values = player_line_values.replace(" ( 0 offensive rebounds )", "")
        paragraph.append(player_line_values)


def get_team_additional_stats(team_city, team_name, fgm, fga, fg_pct, fg3m, fg3a, fg3_pct, ftm, fta, ft_pct, reb, ast,
                              tov, paragraph):
    stats = "%s shot %d - of - %d with %d percent from the field , %d - of - %d with %d percent from three - point " \
            "range and %d - of - %d with %d percent from the free - throw line . %s , as a team added %d assists and %d " \
            "rebounds . The %s were forced into %d turnovers "

    paragraph.append(stats % (
        team_city, fgm, fga, fg_pct, fg3m, fg3a, fg3_pct, ftm, fta, ft_pct, team_city, ast, reb, team_name, tov))


def game_stats_to_text(home_team, away_team):
    first_line = "The %s ( %d - %d ) defeated the %s ( %d - %d ) %d - %d at the %s in %s "
    paragraph = []

    home_city, home_name, home_wins, home_losses, home_pts, home_pts_q1, home_pts_q2, home_pts_q3, home_pts_q4, home_fgm, \
    home_fga, home_fg_pct, home_fg3m, home_fg3a, home_fg3_pct, home_ftm, home_fta, \
    home_ft_pct, home_reb, home_ast, home_tov = get_team_info(home_team.team_stats_dict)
    home_full_name = home_city + " " + home_name

    vis_city, vis_name, vis_wins, vis_losses, vis_pts, vis_pts_q1, vis_pts_q2, vis_pts_q3, vis_pts_q4, vis_fgm, vis_fga, \
    vis_fg_pct, vis_fg3m, vis_fg3a, vis_fg3_pct, vis_ftm, vis_fta, vis_ft_pct, \
    vis_reb, vis_ast, vis_tov = get_team_info(away_team.team_stats_dict)
    vis_full_name = vis_city + " " + vis_name

    home_won = home_pts > vis_pts

    teams_paragraph = []
    if home_won:
        teams_paragraph.append(first_line % (home_city + " " + home_name, home_wins, home_losses,
                                             vis_city + " " + vis_name, vis_wins, vis_losses, home_pts, vis_pts,
                                             home_team.arena_and_location[0], home_team.arena_and_location[1]))
    else:
        teams_paragraph.append(first_line % (vis_city + " " + vis_name, vis_wins, vis_losses,
                                             home_city + " " + home_name, home_wins, home_losses, vis_pts, home_pts,
                                             home_team.arena_and_location[0], home_team.arena_and_location[1]))

    get_quarter_score_line("first", home_full_name, vis_full_name, home_pts_q1, vis_pts_q1, teams_paragraph)
    get_quarter_score_line("second", home_full_name, vis_full_name, home_pts_q2, vis_pts_q2, teams_paragraph)
    get_quarter_score_line("third", home_full_name, vis_full_name, home_pts_q3, vis_pts_q3, teams_paragraph)
    get_quarter_score_line("fourth", home_full_name, vis_full_name, home_pts_q4, vis_pts_q4, teams_paragraph)

    paragraph.append('.'.join(teams_paragraph))
    get_team_additional_stats(home_city, home_name, home_fgm, home_fga, home_fg_pct, home_fg3m, home_fg3a, home_fg3_pct,
                              home_ftm, home_fta, home_ft_pct, home_reb, home_ast, home_tov, paragraph)

    get_team_additional_stats(vis_city, vis_name, vis_fgm, vis_fga, vis_fg_pct, vis_fg3m, vis_fg3a, vis_fg3_pct,
                              vis_ftm, vis_fta, vis_ft_pct, vis_reb, vis_ast, vis_tov, paragraph)

    home_players_paragraph = []
    vis_players_paragraph = []
    get_players_line(home_team.players, home_city, home_players_paragraph)
    get_players_line(away_team.players, vis_city, vis_players_paragraph)

    paragraph.extend(home_players_paragraph)
    paragraph.extend(vis_players_paragraph)
    return paragraph


# ************************************
# ****** summaries preprocess ********
# ************************************

def summaries_cleaner(blacklist, summary):
    clean_summary = []
    i = 0
    sentences = summary.split(".")
    n = len(sentences)
    for sentence in sentences:
        if i == 0 or i == 1 or not is_black_list_in_sent(sentence, blacklist):
            if i == n - 1 and sentence == '\n':
                clean_summary.pop()
                clean_summary.append(sentence)
            elif i != n - 1:
                clean_summary.append(sentence)
            else:
                clean_summary.append('\n')
        else:
            clean_sent = remove_sub_sent(sentence, blacklist)
            if clean_sent:
                clean_summary.append(remove_sub_sent(sentence, blacklist))
        i += 1
    return '.'.join(clean_summary)


def is_black_list_in_sent(sent, blacklist):
    return re.search(blacklist, sent, flags=re.IGNORECASE)


def remove_sub_sent(sent, blacklist):
    clean_sent = []
    for sub_sent in sent.split(","):
        if not is_black_list_in_sent(sub_sent, blacklist):
            clean_sent.append(sub_sent)
    return ','.join(clean_sent)


# ************************************
# *** summaries preprocess end *******
# ************************************
idx = 0


def create_json_format(summary, descriptions):
    global idx
    idx += 1

    return {
        'id': str(idx),
        'summary': summary,
        'descriptions': descriptions
    }


def preprocess_data(input_files, output_file_name, blacklist):
    json_elements = []
    with open(output_file_name, "w") as output_file:
        for input_file_name in input_files:
            with open(input_file_name + "_data.txt", "r", encoding="utf8") as stats_input_file:
                with open(input_file_name + "_text.txt", "r", encoding="utf8") as summaries_input_file:
                    stats = stats_input_file.readlines()
                    summaries = summaries_input_file.readlines()
                    for i in range(len(stats)):
                        home_team, away_team = process_game_stats(stats[i])
                        descriptions = game_stats_to_text(home_team, away_team)
                        summary = summaries_cleaner(blacklist, summaries[i])
                        json_elements.append(create_json_format(summary, descriptions))
        json_element = json.dumps(json_elements, indent=2)
        print(json_element, file=output_file)


if __name__ == '__main__':
    blacklist = 'injured|injury|injuries|referee|refereeing|referees|judge|judges|judging|judgment|all-star|allstar' \
                '|all star|knee|knees|shoulder|shoulders|series|serieses|Achilles|neck|hamstring|Next ,' \
                '|\(.[a-z]+.\)|playoff|playoffs|will head|next game|next games|will travel|road trip|road trips|road-trip '

    input_files = ["./inputs/D1_2014", "./inputs/D1_2015", "./inputs/D1_2016", "./inputs/D1_2017", "./inputs/D1_2018"]
    output_file_name = "./preprocessed_data.json"
    preprocess_data(input_files, output_file_name, blacklist)
