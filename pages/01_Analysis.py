import streamlit as st
import pandas as pd
import pandasql as ps
import kilter_utils
import plotly.express as px
import plotly.graph_objects as go
import math
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
#import kilter_utils
st.set_page_config(layout="wide")

data_path = "C:/Users/Declan/Documents/DataScienceProjects/KilterBoard_project/streamlit_app/data/"

'''# Analysis'''




# location of db file
cnx = kilter_utils.read_in_sqllite_file(data_path)

# create initial df
df = pd.read_sql_query('''SELECT climbs.uuid, climbs.setter_id, climbs.name, climbs.frames, climbs.created_at, climb_stats.angle, climb_stats.display_difficulty, climb_stats.ascensionist_count, climb_stats.difficulty_average, climb_stats.quality_average
                          FROM climbs
                          INNER JOIN climb_stats
                          ON climbs.uuid = climb_stats.climb_uuid
                          WHERE is_listed  = 1 and is_draft = 0 and frames_count = 1 and layout_id = 1 and edge_top <= 152 and edge_left >=4 and edge_right <= 140 and ascensionist_count >=1
                          ''', cnx)

'''#### Filtering database
Limit results to those on the 12x12 board which includes all climbs on the 7x10 and 8x12 boards
Remove unlisted climbs and routes
Finally limit to boulders with at least 5 ascents. This is done to remove climbs which are potentially experimental potentially of lower quality
V grades are used opposed to french/font grades
'''

'''
#
#
#### Climbs by angle
'''
df_climbs_by_angle = ps.sqldf('''SELECT angle, COUNT(*)  AS 'number of climbs'
                                 FROM df GROUP BY angle
                                 ''')

# create figure
fig = px.histogram(df_climbs_by_angle,
                   x="angle",
                   y = 'number of climbs',
                   nbins=df_climbs_by_angle.shape[0]
                  )

fig.update_layout(bargap=0.2,
                  title="Number of climbs at each angle",
                 yaxis_title="Number of climbs",
                 xaxis_title = "Angle")

fig


'''
#
#### Grade skew at each angle '''
df_angle_grade = pd.read_sql_query(''' SELECT board_angle, v_grade, SUM(num_climbs)  AS num_climbs
                                       FROM (
                                            SELECT board_angle, grade, boulder_name, TRIM(substr(boulder_name, instr(boulder_name,'/')+2)) AS v_grade, COUNT(grade) AS num_climbs
                                            FROM (
                                                SELECT climb_stats.angle AS board_angle, round(difficulty_average,0) AS grade
                                                FROM climbs
                                                INNER JOIN climb_stats
                                                ON climbs.uuid = climb_stats.climb_uuid
                                                WHERE is_listed = 1 and is_draft = 0 and frames_count = 1 and layout_id = 1 and edge_top <= 152 and edge_left >=4 and edge_right <= 140) t1
                                                INNER JOIN difficulty_grades
                                                ON difficulty_grades.difficulty = t1.grade
                                                GROUP BY board_angle, grade ORDER BY board_angle ASC, grade ASC)
                                            GROUP BY board_angle, v_grade
                                            ORDER BY board_angle ASC, CAST(v_grade AS int)
                                            ''', cnx)

fig = px.histogram(df_angle_grade, x="v_grade", y="num_climbs", animation_frame="board_angle",
           nbins=25,
           range_y=[0,1000*math.ceil((df_angle_grade['num_climbs'].max()/1000))],range_x=['0','16']
                  )

fig["layout"].pop("updatemenus") # optional, drop animation buttons

fig.update_layout(bargap=0.2,
                  title="Number of climbs at each V grade",
                 yaxis_title="Number of climbs",
                 xaxis_title = "V grade")

fig

################## Heatmap
board_path = f"{data_path}full_board_commercial.png"

df_climbing = pd.read_sql_query('''SELECT uuid, board_angle, frames, quality_average, v_grade
                                   FROM (
                                        SELECT uuid,board_angle, grade, boulder_name, TRIM(substr(boulder_name, instr(boulder_name,'/')+2)) AS v_grade, frames, name, ascensionist_count ,quality_average
                                        FROM (
                                            SELECT uuid, climb_stats.angle AS board_angle, round(difficulty_average,0) AS grade, frames, name, ascensionist_count, quality_average
                                            FROM climbs
                                            INNER JOIN climb_stats
                                            ON climbs.uuid = climb_stats.climb_uuid
                                            WHERE is_listed = 1 and is_draft = 0 and frames_count = 1 and layout_id = 1 and edge_top <= 152 and edge_left >=4 and edge_right <= 140 and frames NOT LIKE '%r2%'and frames NOT LIKE '%r3%') t1
                                            INNER JOIN difficulty_grades
                                            ON difficulty_grades.difficulty = t1.grade
                                            ORDER BY board_angle ASC, grade ASC)
                                        WHERE ascensionist_count >=1  ''', cnx)


st.subheader('What angle has the most ascents?')
num_ascents_by_angle = pd.read_sql_query('''SELECT angle, SUM(ascensionist_count) AS ascents
                                        	FROM(
                                        		SELECT climbs.uuid, climbs.setter_id,climbs.setter_username , climbs.name, climbs.frames, climbs.created_at, climb_stats.angle, climb_stats.display_difficulty, climb_stats.ascensionist_count, climb_stats.difficulty_average, climb_stats.quality_average, climbs.is_listed
                                        		FROM climbs
                                        		INNER JOIN climb_stats
                                        		ON climbs.uuid = climb_stats.climb_uuid
                                        		WHERE is_listed  = 1 and frames_count = 1 and layout_id = 1 and edge_top <= 152 and edge_left >=4 and edge_right <= 140-- and ascensionist_count >= 5
                                        		)
                                        	GROUP BY angle
                                            ''', cnx)
#st.dataframe(num_ascents_by_angle)
fig = px.histogram(num_ascents_by_angle, x="angle", y="ascents",
           nbins=28
           )

fig["layout"].pop("updatemenus") # optional, drop animation buttons

fig.update_layout(bargap=0.2,
                  title="Number of climbs at each angle",
                 yaxis_title="Number of ascents",
                 xaxis_title = "Angle")

fig





st.subheader('What grade has the most ascents?')

num_ascents_by_grade = pd.read_sql_query('''SELECT v_grade, SUM(ascensionist_count) AS ascents
                                            FROM (
                                            	SELECT *, grade, boulder_name, TRIM(substr(boulder_name, instr(boulder_name,'/')+2)) AS v_grade
                                                FROM (
                                                    SELECT *,round(difficulty_average,0) AS grade
                                                    FROM climbs
                                                    INNER JOIN climb_stats
                                                    ON climbs.uuid = climb_stats.climb_uuid
                                                    WHERE is_listed = 1 and is_draft = 0 and frames_count = 1 and layout_id = 1 and edge_top <= 152 and edge_left >=4 and edge_right <= 140 and ascensionist_count >=5) t1
                                                    INNER JOIN difficulty_grades
                                                    ON difficulty_grades.difficulty = t1.grade
                                                    )
                                                GROUP BY v_grade
                                                ORDER BY CAST(v_grade AS int) ASC
                                                ''',cnx )

#st.dataframe(num_ascents_by_grade)

fig = px.histogram(num_ascents_by_grade, x="v_grade", y="ascents",
           nbins=28
           )

fig["layout"].pop("updatemenus") # optional, drop animation buttons

fig.update_layout(bargap=0.2,
                  title="Number of climbs at each grade",
                 yaxis_title="Number of ascents",
                 xaxis_title = "Grade")

fig


st.subheader('Heatmap of climbs ')
st.write("This doesn't differentiate between foot and hand holds, kickboard isn't shown")

if 'angle' not in st.session_state:
    st.session_state['angle'] = 0

if 'grade' not in st.session_state:
    st.session_state['grade'] = 0



try:
    col1,col2,col3 = st.columns([1,3,3])
    with col1:
        angle_options = ['0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60','65','70']
        angle = st.select_slider('Choose an angle', options =  angle_options)

        grade_options = range(0,17)
        grade = st.select_slider('Choose a v grade',  options =  grade_options)

        df_plot = kilter_utils.prepare_hold_heatmap_angle_grade_df(data_path, df_climbing, int(angle), int(grade))

        mainhold_img = kilter_utils.mainholds_heatmap_overlayed(data_path,df_plot)
        auxhold_img  = kilter_utils.aux_holds_heatmap_overlayed(data_path,df_plot)

        st.session_state['angle'] = angle
        st.session_state['grade'] = grade
    with col2:
        st.write('Main holds')
        mainhold_img
    with col3:
        st.write('Auxillary holds')
        auxhold_img
except:
    pass
    #st.session_state['angle'] = angle
    #st.session_state['grade'] = grade

## number of setters
'''
#
## Looking at setters
'''

''' ### How many unique setters are there?'''

num_setters = pd.read_sql_query('''SELECT COUNT(DISTINCT(setter_username))
                                   FROM(
                                    SELECT climbs.uuid, climbs.setter_id,climbs.setter_username , climbs.name, climbs.frames, climbs.created_at, climb_stats.angle, climb_stats.display_difficulty, climb_stats.ascensionist_count, climb_stats.difficulty_average, climb_stats.quality_average
                                    FROM climbs
                                    INNER JOIN climb_stats
                                    ON climbs.uuid = climb_stats.climb_uuid
	                                WHERE is_listed  = 1 and is_draft = 0 and frames_count = 1 and layout_id = 1 and edge_top <= 152 and edge_left >=4 and edge_right <= 140 and ascensionist_count >=1
	                                )''', cnx)
st.write(num_setters)

# which setters have the most ascents
'''
#
### Setters with the most unique climbs - not including drafts or open projects
'''

most_prolific_setters = pd.read_sql_query('''SELECT *,  ROUND(total_ascents/num_climbs,2) as avg_ascents_per_climb
                                             FROM (
    	                                         SELECT setter_username, SUM(ascensionist_count) as total_ascents, COUNT(DISTINCT(uuid)) as num_climbs
                                            	 FROM (
                                            		SELECT climbs.uuid, climbs.setter_id,climbs.setter_username , climbs.name, climbs.frames, climbs.created_at, climb_stats.angle, climb_stats.display_difficulty, climb_stats.ascensionist_count, climb_stats.difficulty_average, climb_stats.quality_average
                                            		FROM climbs
                                            		INNER JOIN climb_stats
                                            		ON climbs.uuid = climb_stats.climb_uuid
                                            		WHERE is_listed  = 1 and is_draft = 0 and frames_count = 1 and layout_id = 1 and edge_top <= 152 and edge_left >=4 and edge_right <= 140
                                            		)
                                            	GROUP BY setter_username
                                                )
                                            	ORDER BY total_ascents DESC
                                                LIMIT 20
                                                ''', cnx	) # and ascensionist_count >=1


#st.dataframe(most_prolific_setters)

fig = px.histogram(most_prolific_setters, x="setter_username", y="num_climbs",
           nbins=28
           ).update_xaxes(categoryorder="total descending")

fig["layout"].pop("updatemenus") # optional, drop animation buttons

fig.update_layout(bargap=0.2,
                  title="Most prolific setters by number of climbs",
                 yaxis_title="Number of climbs",
                 xaxis_title = "Setter")

fig



fig = px.histogram(most_prolific_setters, x="setter_username", y="total_ascents",
           nbins=28
           ).update_xaxes(categoryorder="total descending")

fig["layout"].pop("updatemenus") # optional, drop animation buttons

fig.update_layout(bargap=0.2,
                  title="Most prolific setters by ascents of their climbs",
                 yaxis_title="Number of ascents",
                 xaxis_title = "Setter")
fig.update_xaxes(tickangle=45)
fig

'''
#
### How many unique projects are there?
'''

number_unlisted_projects = pd.read_sql_query('''SELECT COUNT(DISTINCT(uuid)) as num_open_projects
                                            	FROM(
                                            		SELECT climbs.uuid, climbs.setter_id,climbs.setter_username , climbs.name, climbs.frames, climbs.created_at, climb_stats.angle, climb_stats.display_difficulty, climb_stats.ascensionist_count, climb_stats.difficulty_average, climb_stats.quality_average, climbs.is_listed
                                            		FROM climbs
                                            		INNER JOIN climb_stats
                                            		ON climbs.uuid = climb_stats.climb_uuid
                                            		WHERE is_listed  = 0 and frames_count = 1 and layout_id = 1 and edge_top <= 152 and edge_left >=4 and edge_right <= 140-- and ascensionist_count >= 0
                                            		)
                                            	ORDER BY uuid
                                                ''', cnx)
st.dataframe(number_unlisted_projects)

'''
#
### How many open projects across all angles (one climb can be a project at muliple angles)
'''

number_unlisted_projects_multi_angle = pd.read_sql_query('''SELECT COUNT(*) as num_open_projects_multi_angle
                                                        	FROM(
                                                        		SELECT climbs.uuid, climbs.setter_id,climbs.setter_username , climbs.name, climbs.frames, climbs.created_at, climb_stats.angle, climb_stats.display_difficulty, climb_stats.ascensionist_count, climb_stats.difficulty_average, climb_stats.quality_average, climbs.is_listed
                                                        		FROM climbs
                                                        		INNER JOIN climb_stats
                                                        		ON climbs.uuid = climb_stats.climb_uuid
                                                        		WHERE is_listed  = 0 and frames_count = 1 and layout_id = 1 and edge_top <= 152 and edge_left >=4 and edge_right <= 140-- and ascensionist_count >= 0
                                                        		)
                                                                ''', cnx)
st.dataframe(number_unlisted_projects_multi_angle)
