team_name_to_location = {}
team_name_to_location['atlanta hawks'] = ('state farm arena', 'atlanta, georgia')
team_name_to_location['boston celtics'] = ('td garden', 'boston, massachusetts')
team_name_to_location['brooklyn nets'] = ('barclays center', 'brooklyn, new york')
team_name_to_location['charlotte hornets'] = ('spectrum center', 'charlotte, north carolina')
team_name_to_location['chicago bulls'] = ('united center', 'chicago, illinois')
team_name_to_location['cleveland cavaliers'] = ('rocket mortgage fieldhouse', 'cleveland, ohio')
team_name_to_location['dallas mavericks'] = ('american airlines center', 'dallas, texas')
team_name_to_location['denver nuggets'] = ('pepsi center', 'denver, colorado')
team_name_to_location['detroit pistons'] = ('little caesars arena', 'detroit, michigan')
team_name_to_location['golden_state warriors'] = ('chase center', 'san francisco, california')
team_name_to_location['houston rockets'] = ('toyota center', 'houston, texas')
team_name_to_location['indiana pacers'] = ('bankers life fieldhouse', 'indianapolis, indiana')
team_name_to_location['los_angeles clippers'] = ('staples center', 'los angeles, california')
team_name_to_location['los_angeles lakers'] = ('staples center', 'los angeles, california')
team_name_to_location['memphis grizzlies'] = ('fedex forum', 'memphis, tennessee') #hi
team_name_to_location['miami heat'] = ('american airlines arena', 'miami, florida')
team_name_to_location['milwaukee bucks'] = ('fiserv forum', 'milwaukee, wisconsin')
team_name_to_location['minnesota timberwolves'] = ('target center', 'minneapolis, minnesota')
team_name_to_location['new_orleans pelicans'] = ('smoothie king center', 'new orleans, louisiana')
team_name_to_location['new_york knicks'] = ('madison square garden', 'new york city, new york')
team_name_to_location['oklahoma_city thunder'] = ('chesapeake energy arena', 'oklahoma city, oklahoma')
team_name_to_location['orlando magic'] = ('amway center', 'orlando, florida')
team_name_to_location['philadelphia 76ers'] = ('wells fargo center', 'philadelphia, pennsylvania')
team_name_to_location['phoenix suns'] = ('talking stick resort arena', 'phoenix, arizona')
team_name_to_location['portland trail_blazers'] = ('moda center', 'portland, oregon')
team_name_to_location['sacramento kings'] = ('golden 1 center', 'sacramento, california')
team_name_to_location['san_antonio spurs'] = ('at&t center', 'san antonio, texas')
team_name_to_location['toronto raptors'] = ('scotiabank arena', 'toronto, ontario')
team_name_to_location['utah jazz'] = ('vivint smart home arena', 'salt lake city, utah')
team_name_to_location['washington wizards'] = ('capital one arena', 'washington, d.c.')


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
        self.players = [player for player in self.players
                        if int(player.player_stats_dict['MIN']) > 3 or player.player_stats_dict['STARTER'] == 'yes']

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


def process_game(game_data):
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


def preprocess_data(input_file_name, output_file_name):
    with open(input_file_name, "r", encoding="utf8") as input_file:
        with open(output_file_name, "w") as output_file:
            for line in input_file:
                home_team, away_team = process_game(line)
                output_file.write(game_stats_to_text(home_team, away_team))


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
    fg_pct = float(team_dict["TEAM-FG_PCT"])
    fg3m = int(team_dict["TEAM-FG3M"])
    fg3a = int(team_dict["TEAM-FG3A"])
    fg3_pct = float(team_dict["TEAM-FG3_PCT"])
    ftm = int(team_dict["TEAM-FTM"])
    fta = int(team_dict["TEAM-FTA"])
    ft_pct = float(team_dict["TEAM-FT_PCT"])
    reb = int(team_dict["TEAM-REB"])
    ast = int(team_dict["TEAM-AST"])
    tov = int(team_dict["TEAM-TOV"])

    return city, name, wins, losses, pts, pts_q1, pts_q2, pts_q3, pts_q4, fgm, fga, fg_pct, fg3m, fg3a, fg3_pct, ftm, fta, ft_pct, reb, ast, tov


def game_stats_to_text(home_team, away_team):
    first_line = "The %s ( %d - %d ) defeated the %s ( %d - %d ) %d - %d at the %s in %s."
    player_line = "%s scored %d points ( %d - %d FG , %d - %d 3Pt , %d - %d FT ) to go with %d rebounds ."

    paragraph = []

    home_city, home_name, home_wins, home_losses, home_pts, home_pts_q1, home_pts_q2, home_pts_q3, home_pts_q4, home_fgm, \
    home_fga, home_fg_pct, home_fg3m, home_fg3a, home_fg3_pct, home_ftm, home_fta, \
    home_ft_pct, home_reb, home_ast, home_tov = get_team_info(home_team.team_stats_dict)

    vis_city, vis_name, vis_wins, vis_losses, vis_pts, vis_pts_q1, vis_pts_q2, vis_pts_q3, vis_pts_q4, vis_fgm, vis_fga, \
    vis_fg_pct, vis_fg3m, vis_fg3a, vis_fg3_pct, vis_ftm, vis_fta, vis_ft_pct, \
    vis_reb, vis_ast, vis_tov = get_team_info(away_team.team_stats_dict)

    home_won = home_pts > vis_pts
    if home_won:
        paragraph.append(first_line % (home_city + " " + home_name, home_wins, home_losses,
                                       vis_city + " " + vis_name, vis_wins, vis_losses, home_pts, vis_pts,
                                       home_team.arena_and_location[0], home_team.arena_and_location[1]))
    else:
        paragraph.append(first_line % (vis_city + " " + vis_name, vis_wins, vis_losses,
                                       home_city + " " + home_name, home_wins, home_losses, vis_pts, home_pts,
                                       home_team.arena_and_location[0], home_team.arena_and_location[1]))
    return ''.join(paragraph)


if __name__ == '__main__':
    input_file_name = "./D1_2014_data.txt"
    output_file_name = "./D1_2014_data_out.txt"
    preprocess_data(input_file_name, output_file_name)
