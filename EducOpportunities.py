import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import warnings
import folium
from streamlit_folium import folium_static
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from math import pi
warnings.filterwarnings('ignore')

##### NOTE ####
# Replace the shapefile link (the one we used when discussing geopandas) on pages 4 (Insight on MOOE) and 5 (Insight on School Rooms) based on where it is in your pc #


#-----Start of Set Up-----#

df_all = pd.read_csv("./education_analytics/masterfile.csv")
shapefile = gpd.read_file('./education_analytics/PH Provinces Shapefile/Provinces.shp')
df_map = df_all[df_all['Longitude']>100]
df_features = df_all[["student_per_rm","mooe_student_ratio","student_teacher_ratio"]]

my_page = st.sidebar.radio('Contents',['Introduction','Data Information','Methodology', 'Summary of Formed Clusters','Engineered Features','Cluster Map','Other Cluster Insights','Conclusions and Recommendations']) # creates sidebar #

st.markdown("""<style>.css-1aumxhk {background-color: #ebeae3;background-image: none;color: #ebeae3}</style>""", unsafe_allow_html=True) # changes background color of sidebar #

#-----End of Set Up-----#


#-----Start of Page 1 (Introduction)-----#

if my_page == 'Introduction':

    st.title("Allocation of Resources for School Congestion in Elementary Schools")
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between title and paragraph #
    st.markdown("To achieve accessible, relevant, and liberating basic education for all, the Philippine Development Plan 2017-2022 identifies the following as part of the country’s core strategies: (1) Increasing investments  in education to improve quality, and (2) Extending opportunities to those outside of the formal education.")
    c1, c2 = st.beta_columns(2)
    image1 = 'https://www.theywill.dance/wp-content/uploads/2021/01/5240.jpg' # hosted image on my website #
    image2 = 'https://www.theywill.dance/wp-content/uploads/2021/01/5236.jpg' # hosted image on my website #
    c1.markdown('<div style="text-align: center;color: #F7A92D; font-size: large;font-weight: bold;">Increasing Investments</div>',unsafe_allow_html=True)
    c1.image(image1,use_column_width=True)
    c2.markdown('<div style="text-align: center;color: #A6C3D0; font-size: large;font-weight: bold;">Extending Opportunities</div>',unsafe_allow_html=True)
    c2.image(image2,use_column_width=True)
    st.markdown("We examined initial indicators available through the public data from the Department of Education that may provide context  to these two strategies.")
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between title and subheader #
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between title and subheader #
    st.markdown('<div style="color: #F7A92D; font-size: medium;font-weight: bold;">What we wanted to know</div>',unsafe_allow_html=True)
    st.subheader("How can the Department of Education allocate their resources to address School Congestion in elementary schools?")
#-----End of Page 1 (Introduction)-----#


#-----Start of Page 2 (Data Information)-----#

elif my_page == 'Data Information':
    st.title("Allocation of Resources for School Congestion in Elementary Schools")
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between title and subheader #
    st.subheader("Data Information")
    image3 = 'https://www.theywill.dance/wp-content/uploads/2021/01/Sprin1_Lab_Presentation-Team-Pat.png' # hosted image on my website #
    st.image(image3,use_column_width=True)
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between image and paragraph #
    st.markdown("We used public data from the Department of Education dated 2015. The seven datasets we used provided information on the masterlist of schools and their number of teachers, Maintenance and Other Operating Expenses (MOOE),  rooms, location, and enrollees, as well as enrollee information like gender, grade level, and class type (regular and SPED).")
    
#-----End of Page 2 (Data Information)-----#


#-----Start of Page 3 (Methodology)-----#

elif my_page == 'Methodology':
    st.title("Allocation of Resources for School Congestion in Elementary Schools")
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between title and subheader #
    st.subheader("Methodology")
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between subheader and image #
    image4 = 'https://www.theywill.dance/wp-content/uploads/2021/01/Sprin1_Lab_Presentation-Team-Pat-2.png' # hosted image on my website #
    st.image(image4,use_column_width=True)
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between image and paragraph #
    st.markdown("We created a master data where we merged all of the seven datasets so that we were using a single version of the master data, considering that each of us are working on each of the questions we wanted to explore.")
    st.markdown("After having a master data, we wrangled it into different data sets that would suit in answering the questions that we have.")
    st.markdown("We then proceeded to create the data visualisations based on the datasets that we created.")

#-----End of Page 3 (Methodology)-----#

#-----Start of Page 4 (Summary of Formed Clusters)-----#

elif my_page == 'Summary of Formed Clusters':
    st.title("Allocation of Resources for School Congestion in Elementary Schools")
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between title and subheader #
    st.subheader("Summary of Formed Clusters")
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between subheader and image #
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between subheader and image #
    
    df_features = df_features.rename(columns={'mooe_student_ratio' : 'MOOE Per Student Ratio','student_per_rm' : 'Student Per Room Ratio','student_teacher_ratio':'Student/Teacher Ratio'})
    
    # Pre-processing
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_features)
    
    # Getting the clusters with K-means
    model = KMeans(n_clusters=3, random_state = 42)
    model.fit(df_scaled)
    cluster_labels = model.predict(df_scaled)
    
    #Prep for Radar Chart
    
    scaler = MinMaxScaler()
    df_minmax = scaler.fit_transform(df_features)

    df_minmax = pd.DataFrame(df_minmax, index=df_features.index, columns=df_features.columns)
    df_minmax['clusters'] = cluster_labels

    df_clusters = df_minmax.set_index("clusters")
    df_clusters = df_clusters.groupby("clusters").mean().reset_index()
    
    #Radar function
    
    def make_spider( row, title, color):

        # number of variable
        categories=list(df_clusters)[1:]
        N = len(categories)

        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        # Initialise the spider plot
        ax = plt.subplot(3,3,row+1, polar=True )

        # If you want the first axis to be on top:
        ax.set_theta_offset(pi / 3.5)
        ax.set_theta_direction(-1)

        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles[:-1], categories, color='black', size=8)

        # Draw ylabels
        ax.set_rlabel_position(0)

        plt.yticks([-0.25, 0, 0.25, 0.5, 0.75, 1], [-0.25, 0, 0.25, 0.5,0.75, 1], color="grey", size=7) #formmscaled
        plt.ylim(-0.25,1)

        # Ind1
        values=df_clusters.loc[row].drop('clusters').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
        ax.fill(angles, values, color=color, alpha=0.4)

        # Add a title
        plt.title(title, size=12, color=color, y=1.1)
        
    #---- Plotting the Radar Chart----
    my_dpi=200
    fig = plt.figure(figsize=(2200/my_dpi, 2000/my_dpi), dpi=my_dpi)
    plt.subplots_adjust(hspace=0.5)


    # Create a color palette:
    my_palette = plt.cm.get_cmap("Set3_r", len(df_clusters.index))

    for row in range(0, len(df_clusters.index)):
        make_spider(row=row, 
                    title='Cluster '+(df_clusters['clusters'][row]+1).astype(str), 
                    color=my_palette(row))
    plt.show()
    # end of generating map #
    
    st.pyplot(fig) # show graph #
    # start of insights #
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between graph and insights #
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between graph and insights #
    st.markdown('<div style="color: #FCCF55; font-size: medium;padding-bottom: 15px">Insights</div>',unsafe_allow_html=True)
    st.markdown(':bulb:  Cluster 1 - Schools with low demand & high resources')
    st.markdown('      - Mostly situated in rural areas')
    st.markdown('      - Low student-teacher ratio')
    st.markdown('      - Low student per room ratio')
    st.markdown('      - High MOOE per student')
    st.markdown(':bulb:  Cluster 2 - Schools with high demand & low resources')
    st.markdown('      - Mostly situated in urban areas')
    st.markdown('      - High student-teacher ratio')
    st.markdown('      - High student per room ratio')
    st.markdown('      - Low MOOE per student')
    st.markdown(':bulb:  Cluster 3 - Schools with moderate demand & resources')
    st.markdown('      - Found in both urban/rural areas')
    st.markdown('      - Moderate student-teacher ratio')
    st.markdown('      - Moderate student per room ratio')
    st.markdown('      - Moderate MOOE per student')
    # end of insights #
    
#-----End of Page 4 (Summary of Formed Clusters)-----#

#-----Start of Page 5 (Engineered Features Map)-----#

elif my_page == 'Engineered Features':
    
    st.title("Allocation of Resources for School Congestion in Elementary Schools")
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between title and subheader #
    st.subheader("Mapping the Engineered Features per Province")
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between subheader and graph #
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between subheader and graph #   
    
    # Create dictionary of those with discrepancy
    province_dic = {'CITY OF COTABATO':'Maguindanao',
     'Manila, Ncr, First District':"Metropolitan Manila",
     'Ncr Fourth District':"Metropolitan Manila",
     'Ncr Second District':"Metropolitan Manila",
     'Ncr Third District':"Metropolitan Manila",
     'Western Sama':"Samar"}
    
    # Replace province name
    df_all["school.province"] = df_all["school.province"].str.title().replace(province_dic).str.replace("Del", 'del')
    
    # for mapping
    df_st   = df_all.groupby("school.province")["student_teacher_ratio"].mean().reset_index()
    df_mooe = df_all.groupby("school.province")["mooe_student_ratio"].mean().reset_index()
    df_sr   = df_all.groupby("school.province")["student_per_rm"].mean().reset_index()
    
    # merging the df's for mapping with shapefile
    merged_data = pd.merge(shapefile, df_st, left_on='PROVINCE', right_on="school.province")
    merged_data = merged_data.merge(df_mooe)
    merged_data = merged_data.merge(df_sr)
    
    #Setting Variables
    var1 = 'student_teacher_ratio'
    var2 = 'student_per_rm'
    var3 = 'mooe_student_ratio'

    # set the range for the choropleth
    vmin1, vmax1 = merged_data['student_teacher_ratio'].min(), merged_data['student_teacher_ratio'].max()
    vmin2, vmax2 = merged_data["student_per_rm"].min(), merged_data["student_per_rm"].max()
    vmin3, vmax3 = merged_data["mooe_student_ratio"].min(), merged_data["mooe_student_ratio"].max()

    fig, axes = plt.subplots(1,3, figsize=(18, 10))

    #st ratio
    merged_data.plot(column=var1, cmap='Oranges', linewidth=0.8, ax=axes[0], edgecolor='0.8', vmin=vmin1, vmax=vmax1)
    sm = plt.cm.ScalarMappable(cmap='Oranges', norm=plt.Normalize(vmin=vmin1, vmax=vmax1))
    cbar = fig.colorbar(sm, ax=axes[0], shrink = 0.72)
    axes[0].set_title("Average Student-Teacher Ratio per Province",fontsize=12,y=1.02)
    
    #sr ratio
    merged_data.plot(column=var2, cmap='Oranges', linewidth=0.8, ax=axes[1], edgecolor='0.8', vmin=vmin2, vmax=vmax2)
    sm = plt.cm.ScalarMappable(cmap='Oranges', norm=plt.Normalize(vmin=vmin2, vmax=vmax2))
    cbar = fig.colorbar(sm, ax=axes[1], shrink = 0.72)
    axes[1].set_title("Average Student per Room Ratio per Province",fontsize=12,y=1.02)
   
    #mooe ratio
    merged_data.plot(column=var3, cmap='Oranges', linewidth=0.8, ax=axes[2], edgecolor='0.8', vmin=vmin3, vmax=vmax3)
    sm = plt.cm.ScalarMappable(cmap='Oranges', norm=plt.Normalize(vmin=vmin3, vmax=vmax3))
    cbar = fig.colorbar(sm, ax=axes[2], shrink = 0.72)
    axes[2].set_title("Average MOOE per Student per Province",fontsize=12,y=1.02)
    # end of generating map #
    
    st.pyplot(fig) # show graph #
    
    # start of insights #
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between graph and insights #
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between graph and insights #
    st.markdown('<div style="color: #FCCF55; font-size: medium;padding-bottom: 15px">Insights</div>',unsafe_allow_html=True)
    st.markdown(':bulb:  Provinces like Rizal, Bulacan, and Cebu have the highest average student-teacher ratio; while most of the regions in Luzon like CAR, Region I and II enjoys the lowest average student-teacher ratio.')
    st.markdown(':bulb:  High student room ratio in NCR, South Luzon, and Mindanao; while lower student room ratio in North Luzon')
    st.markdown(':bulb:  High MOOE per Student in Northern Luzon, Samar, Zamboanga; while low MOOE per Student in Southern Luzon, and the regions around Cotabato')
    # end of insights #
    
#-----End of Page 5 (Engineered Features Map)-----#


#-----Start of Page 6 (Cluster Map)-----#

elif my_page == 'Cluster Map':
    
    option = st.sidebar.selectbox(
        'Which cluster do you want to see?',
         df_all['cluster'].unique())

    'You selected: ', option
    
    st.title("Allocation of Resources for School Congestion in Elementary Schools")
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between title and subheader #
    #st.markdown('<div style="color: #F7A92D; font-size: medium;font-weight: bold;">Increasing Investments</div>',unsafe_allow_html=True)
    st.subheader("Mapping schools belonging to the different clusters")
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space subheader and graph #
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space subheader and graph #
    
    from shapely.geometry import Point, Polygon

    geometry = [Point(xy) for xy in zip(df_map['Longitude'], df_map['Latitude'])]
    geo_df = gpd.GeoDataFrame(df_map, geometry = geometry)
    
    if option == 0:
        color = '#f1c232'
    elif option == 1:
        color = '#e06666'
    else:
        color = '#45818e'
        
    fig, ax = plt.subplots(figsize=(15, 15))
    shapefile.plot(ax=ax, alpha = 0.4, color='grey')
    geo_df[geo_df['cluster']==option].plot(ax=ax, marker='+',color=color, markersize=8)
    plt.title("Map of Schools in Cluster " + str(option) ,fontsize=20)
    plt.show()
    # end of generating map #
    
    st.pyplot(fig) # show graph #
    
    # start of insights #
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between graph and insights #
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between graph and insights #
    st.markdown('<div style="color: #FCCF55; font-size: medium;padding-bottom: 15px">Insights</div>',unsafe_allow_html=True)
    if option == 0:
        st.markdown(':bulb:  Upper Luzon (e.g. Isabela, Iloilo, Pangasinan), Western & Eastern Visayas (e.g. Leyte, Bohol). Least common in NCR.')
    elif option == 1:
        st.markdown(':bulb:  Central to South Luzon (e.g. CALABARZON, Bulacan), Central Visayas (e.g. Cebu). Most common in NCR')
    else:
        st.markdown(':bulb:  Upper Luzon (e.g. Pangasinan, Iloilo, Isabela), Western & Eastern Visayas (e.g. Leyte, Negros Occidental)')
    # end of insights #
    
#-----End of Page 6 (Cluster Map)-----#


#-----Start of Page 7 (Other Cluster Insights)-----#

elif my_page == 'Other Cluster Insights':
    
    st.title("Allocation of Resources for School Congestion in Elementary Schools")
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between title and subheader #
    st.subheader("Majority of Elementary Schools in the Philippines are managed by DepEd ")
    st.markdown('<div style="color: #A6C3D0; font-size: medium;font-weight: bold;">School Management Classification per Cluster</div>',unsafe_allow_html=True)
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space subheader and graph #

    
    # School Management Clusters
    df_cluster_feat = df_all[["student_per_rm","mooe_student_ratio","student_teacher_ratio", "school.classification2", 
                            "school.organization", "cluster"]]
    df_class2 = df_cluster_feat.groupby('cluster')['school.classification2'].value_counts().to_frame()
    df_class2 = df_class2.rename(columns={'school.classification2' : 'total_count'})
    df_class2 = df_class2.reset_index()
    
    fig = plt.figure(figsize=(8,6))
    sns.set_theme(style="whitegrid", palette="muted")
    sns.barplot(x='cluster', y='total_count', hue='school.classification2', data=df_class2)
    plt.xlabel('Cluster')
    plt.ylabel("count")
    plt.xticks(ticks = (0, 1, 2), labels = ['1', '2', '3'])
    
    st.pyplot(fig) # show graph #
    
    # start of insights #
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between graph and insights #
    st.markdown('<div style="color: #FCCF55; font-size: medium;padding-bottom: 15px">Insights</div>',unsafe_allow_html=True)
    st.markdown(':bulb:  All clusters shows majority are managed by DepEd.')
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between graph and insights #
    
    # School Organization Clusters
    st.subheader("Majority of Elementary Schools are in Monograde Classroom Set-up")
    st.markdown('<div style="color: #A6C3D0; font-size: medium;font-weight: bold;">School Organization per Cluster</div>',unsafe_allow_html=True)
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space subheader and graph #
    df_org = df_cluster_feat.groupby('cluster')['school.organization'].value_counts().to_frame()
    df_org = df_org.rename(columns={'school.organization' : 'count'})
    df_org = df_org.reset_index()
    
    fig = plt.figure(figsize=(8,6))
    sns.set_theme(style="whitegrid", palette="muted")
    sns.barplot(x='cluster', y='count', hue='school.organization', data=df_org)
    plt.xlabel('Cluster')
    plt.ylabel("count")
    plt.xticks(ticks = (0, 1, 2), labels = ['1', '2', '3'])
    
    st.pyplot(fig) # show graph #
    
    # start of insights #
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between graph and insights #
    st.markdown('<div style="color: #FCCF55; font-size: medium;padding-bottom: 15px">Insights</div>',unsafe_allow_html=True)
    st.markdown(':bulb:  Cluster 1 has close numbers of Monograde and Combined type of classroom setup.')
    st.markdown(':bulb:  Cluster 1 has also the highest count of Multigrade classroom setup.')
    st.markdown(':bulb:  Cluster 3 has high numbers of Monograde classroom setup.')
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between graph and insights #
    
    # SPED Clusters
    st.subheader("SPED opportunities are present in clusters where schools are mostly in urban locations.")
    st.markdown('<div style="color: #A6C3D0; font-size: medium;font-weight: bold;">Incidence of SPED Enrollees per Cluster</div>',unsafe_allow_html=True)
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space subheader and graph #
    spedcount=df_all.groupby("cluster")["total_enrollees_sped"].sum().to_frame().reset_index()
    spedcount.rename(columns={ spedcount.columns[0]: "Cluster" }, inplace = True)
    spedcount.rename(columns={ spedcount.columns[1]: "No. of SPED Enrollees" }, inplace = True)
    
    fig = plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid", palette="muted")
    sns.barplot(x="Cluster", y="No. of SPED Enrollees", data=spedcount)
    plt.xticks(ticks = (0, 1, 2), labels = ['1', '2', '3'])
    
    st.pyplot(fig) # show graph #
    
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between graph and insights #
    st.markdown('<div style="color: #FCCF55; font-size: medium;padding-bottom: 15px">Insights</div>',unsafe_allow_html=True)
    st.markdown(':bulb:  Cluster 2 has the highest incident of SPED enrollees, which can be associated with the highly urbanized location of schools.')
    # end of insights #
    
#-----End of Page 7 (Other Cluster Insights)-----#


#-----Start of Page 8 (Conclusions and Recommendations)-----#

elif my_page == 'Conclusions and Recommendations':
    
    st.title("Allocation of Resources for School Congestion in Elementary Schools")
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between title and subheader #
    st.subheader("Conclusions and Recommendations (Data Insights)")
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between subheader and paragraph #
    image5 = 'https://www.unicef.org/philippines/sites/unicef.org.philippines/files/styles/hero_desktop/public/UNIPH2019008.jpg?itok=CnZKe5zQ'
    st.image(image5,use_column_width=True)
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between subheader and paragraph #
    st.markdown(':bulb:  Consider giving more resources to urban areas given their higher school populations')
    st.markdown(':bulb:  Examine reasons for low student enrollment in rural areas')
    st.markdown('  - Consider examining other factors that affect access to schools (e.g. transportation, economic situation of families, etc) that may prevent students from going to schools - that may lead to underutilization of  resources')
    st.markdown(':bulb:  Examine demand for SPED in schools in rural areas')
    st.markdown('  - For areas which do not have SPED offerings, where do students who need SPED go to?')
    
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between title and subheader #
    st.subheader("Conclusions and Recommendations (Machine Learning)")
    st.markdown(':bulb:  Look into other features which may better categorize schools and improve clustering statistics like the inertia score and silhouette score.')
    st.markdown(':bulb:  Experiment with the different hyperparameters for K-Means and Hierarchical Clustering (init, distance metrics, linkages). Also check if there are other algorithms that may result to better clustering.')
#-----End of Page 8 (Conclusion and Recommendation)-----#