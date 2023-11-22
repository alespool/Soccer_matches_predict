# league and countries
leagues_country_query = """
SELECT 
       id,
       Substr(NAME, 1, Instr(NAME, ' ') - 1) AS first_word,
       Substr(NAME, Instr(NAME, ' ') + 1)    AS second_word
FROM   league 
"""

# how many teams in each league
teams_per_league = """
SELECT c.NAME AS Country,
       Count(DISTINCT( team_long_name )) AS 'No. of Teams'
FROM   match AS m
       LEFT JOIN country AS c
              ON m.country_id = c.id
       LEFT JOIN team AS t
              ON m.home_team_api_id = t.team_api_id
GROUP  BY country 
"""
# number of goals for each league ascending order
league_num_goals_query = """
SELECT 
    Substr(League.name, 1, Instr(League.name, ' ') - 1) AS Country,
    Substr(League.name, Instr(League.name, ' ') + 1) AS League, 
    SUM(home_team_goal + away_team_goal) AS TotalGoals
FROM Match
JOIN League ON Match.league_id = League.id
GROUP BY League.name
ORDER BY TotalGoals DESC;
"""

# goals scored by each team grouped by season
teams_goals_query = """
SELECT c.NAME                AS country,
       t.team_long_name      AS team,
       Sum(m.home_team_goal) AS home_goal,
       Sum(m.away_team_goal) AS away_goal
FROM   match AS m
       LEFT JOIN country AS c
              ON m.country_id = c.id
       LEFT JOIN team AS t
              ON m.home_team_api_id = t.team_api_id
GROUP  BY country,
          team
ORDER  BY home_goal DESC;
"""

# most successful teams compared for all seasons
teams_win_percentage = """
SELECT
    team.team_long_name AS Team,
    SUM(total_games_played_for_the_season) AS TotalGames,
    SUM(number_of_wins) AS Wins,
    ROUND(100.0 * SUM(number_of_wins) / SUM(total_games_played_for_the_season), 4) AS WinPercentage
FROM
    (SELECT
        m.season,
        t.team_long_name,
        COUNT(m.match_api_id) AS total_games_played_for_the_season,
        SUM(CASE
            WHEN (m.home_team_api_id = t.team_api_id AND m.home_team_goal > m.away_team_goal) OR
                 (m.away_team_api_id = t.team_api_id AND m.away_team_goal > m.home_team_goal) THEN 1
            ELSE 0
        END) AS number_of_wins
    FROM match m
    JOIN team t ON m.home_team_api_id = t.team_api_id OR m.away_team_api_id = t.team_api_id
    GROUP BY m.season, t.team_long_name) AS team_stats
JOIN team ON team_stats.team_long_name = team.team_long_name
GROUP BY team.team_long_name
ORDER BY WinPercentage DESC;
"""

# sorting players by potential in descending order
players_potential_query = """
SELECT
       p.player_name AS Player_Name,
       strftime('%d-%m-%Y', p.birthday) AS DOB,
       (strftime('%Y', 'now') - strftime('%Y', p.birthday) - 6) AS Age,
       MAX(p.height) AS Height,
       MAX(p.weight) AS Weight,
       MAX(pa.overall_rating) AS Rating,
       MAX(pa.potential) AS Potential,
       pa.preferred_foot AS Preferred_Foot,
       pa.attacking_work_rate AS Attacking_Work_Rate,
       pa.defensive_work_rate AS Defensive_Work_Rate
FROM player p
JOIN player_attributes pa ON p.player_api_id = pa.player_api_id AND p.player_fifa_api_id = pa.player_fifa_api_id
WHERE
       p.player_name IS NOT NULL AND
       p.birthday IS NOT NULL AND
       p.height IS NOT NULL AND
       p.weight IS NOT NULL AND
       pa.overall_rating IS NOT NULL AND
       pa.potential IS NOT NULL AND
       pa.preferred_foot IS NOT NULL AND
       pa.attacking_work_rate IS NOT NULL AND
       pa.defensive_work_rate IS NOT NULL
GROUP BY p.player_name
ORDER BY Potential DESC
"""

avg_potentials = """
SELECT CASE
         WHEN Round(height) < 165 THEN 165
         WHEN Round(height) > 195 THEN 195
         ELSE Round(height)
       END                                    AS calc_height,
       Count(height)                          AS distribution,
       ( Avg(PA_Grouped.avg_overall_rating) ) AS avg_overall_rating,
       ( Avg(PA_Grouped.avg_potential) )      AS avg_potential,
       Avg(weight)                            AS avg_weight
FROM   player
       LEFT JOIN (SELECT player_attributes.player_api_id,
                         Avg(player_attributes.overall_rating) AS
                         avg_overall_rating,
                         Avg(player_attributes.potential)      AS avg_potential
                  FROM   player_attributes
                  GROUP  BY player_attributes.player_api_id) AS PA_Grouped
              ON player.player_api_id = PA_Grouped.player_api_id
GROUP  BY calc_height
ORDER  BY calc_height 
"""

match_analysis = """
SELECT
    m.league_id,
    c.NAME AS country,
    Substr(l.NAME, Instr(l.NAME, ' ') + 1) AS league,
    m.season,
    m.date,
    m.match_api_id,
    m.home_team_api_id,
    m.away_team_api_id,
    m.home_team_goal AS home_goal,
    m.away_team_goal AS away_goal,
    m.home_player_1,
    m.home_player_2,
    m.home_player_3,
    m.home_player_4,
    m.home_player_5,
    m.home_player_6,
    m.home_player_7,
    m.home_player_8,
    m.home_player_9,
    m.home_player_10,
    m.home_player_11,
    m.away_player_1,
    m.away_player_2,
    m.away_player_3,
    m.away_player_4,
    m.away_player_5,
    m.away_player_6,
    m.away_player_7,
    m.away_player_8,
    m.away_player_9,
    m.away_player_10,
    m.away_player_11,
    m.goal AS goals,
    m.shoton,
    m.shotoff,
    m.foulcommit,
    m.card,
    m.cross,
    m.corner,
    m.possession,
    m.B365H,
    m.B365D,
    m.B365A,
    m.BWH,
    m.BWD,
    m.BWA,
    m.IWH,
    m.IWD,
    m.IWA,
    m.LBH,
    m.LBD,
    CASE
        WHEN m.home_team_goal < m.away_team_goal THEN 0
        WHEN m.home_team_goal = m.away_team_goal THEN 1
        WHEN m.home_team_goal > m.away_team_goal THEN 2
    END AS outcome
FROM
    match m
JOIN
    country c ON m.country_id = c.id
JOIN
    league l ON m.league_id = l.id
JOIN
    team home_team ON m.home_team_api_id = home_team.team_api_id
JOIN
    team away_team ON m.away_team_api_id = away_team.team_api_id
WHERE 1=1
AND country = 'England'
AND league = 'Premier League'
"""

players_stats_query = """
SELECT
    p.player_name,
    pa.player_fifa_api_id,
    pa.player_api_id,
    pa.id,
    (strftime('%Y', 'now') - strftime('%Y', p.birthday) - 6) AS Age,
    pa.overall_rating,
    pa.preferred_foot,
    pa.attacking_work_rate,
    pa.defensive_work_rate,
    pa.acceleration,
    pa.ball_control,
    pa.reactions,
    pa.stamina,
    pa.strength,
    pa.aggression
FROM Player_Attributes pa
JOIN Player p ON p.id = pa.id;
"""

team_stats_query = """
SELECT
    t.team_long_name as Team,
    c.name AS Country,
    Substr(l.NAME, Instr(l.NAME, ' ') + 1)  AS league,
    ta.id,
    ta.team_fifa_api_id,
    ta.team_api_id,
    ta.defencePressure,
    ta.defenceAggression,
    ta.buildUpPlaySpeed,
    ta.buildUpPlayPassing
FROM Team_Attributes ta
JOIN Team t ON t.team_fifa_api_id = ta.team_fifa_api_id
JOIN Match m ON m.away_team_api_id = t.team_api_id
JOIN Country c ON c.id = m.country_id
JOIN League l ON l.id = m.league_id
WHERE 1=1
AND Country = 'England'
AND League = 'Premier League'
GROUP BY Team;
"""