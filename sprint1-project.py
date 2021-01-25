import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')

##### NOTE ####
# Replace the shapefile link (the one we used when discussing geopandas) on pages 4 (Insight on MOOE) and 5 (Insight on School Rooms) based on where it is in your pc #


#-----Start of Set Up-----#

df_all = pd.read_csv("./education_analytics/masterfile.csv")
df_features = df_all[["student_per_rm","mooe_student_ratio","student_teacher_ratio"]]
shapefile = gpd.read_file('./education_analytics/PH Provinces Shapefile/Provinces.shp')
df_map = df_all[df_all['Longitude']>100]

my_page = st.sidebar.radio('Contents',['Introduction','Data Information','Methodology', 'EDA','Cluster Analysis','Other Cluster Insights','Conclusions and Recommendations']) # creates sidebar #

st.markdown("""<style>.css-1aumxhk {background-color: #ebeae3;background-image: none;color: #ebeae3}</style>""", unsafe_allow_html=True) # changes background color of sidebar #

#-----End of Set Up-----#


#-----Start of Page 1 (Introduction)-----#

if my_page == 'Introduction':

    st.title("Allocation of Resources for School Congestion in Elementary Schools")
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between title and paragraph #
    st.markdown("To achieve accessible, relevant, and liberating basic education for all, the Philippine Development Plan 2017-2022 identifies the following as part of the country’s core strategies: (1) Increasing investments  in education to improve quality, and (2) Extending opportunities to those outside of the formal education.")
    c1, c2 = st.beta_columns(2)
    c1.markdown('<div style="text-align: center;color: #F7A92D; font-size: large;font-weight: bold;">Increasing Investments</div>',unsafe_allow_html=True)
    c1.image('./images/increasinginvestments.png',use_column_width=True)
    c2.markdown('<div style="text-align: center;color: #A6C3D0; font-size: large;font-weight: bold;">Extending Opportunities</div>',unsafe_allow_html=True)
    c2.image('./images/extendingopportunities.png',use_column_width=True)
    st.markdown("As part of the strategy of increasing investments in education, we wanted to know how the Department of Education can target their efforts and resources to address congestion in elementary schools.")
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between paragraph and image #
    st.image('./images/overpop.png',use_column_width=True)
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between paragraph and image #
    st.markdown("But school congestion is a nuanced and complex concept, which cannot be capture by a single indicator alone. So we used three indicators such as the (1) student-teacher ratio, (2) student per school room ratio, and (3) MOOE per student. Using these indicators as features, we performed a clustering algorithm on our data to come up with clusters, which then could help the Department of Education in focusing their resources on clusters that may be in need of those resources more.")
#-----End of Page 1 (Introduction)-----#

#-----Start of Page 2 (Data Information)-----#

elif my_page == 'Data Information':
    
    st.title("Data Information")
    st.image('./images/datainfo1.png',use_column_width=True)
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between image and paragraph #
    st.markdown("We used public data from the Department of Education dated 2015. The seven datasets we used provided information on the masterlist of schools and their number of teachers, Maintenance and Other Operating Expenses (MOOE),  rooms, location, and enrollees, as well as enrollee information like gender, grade level, and class type (regular and SPED).")
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between image and paragraph #
    st.image('./images/datainfo2.png',use_column_width=True)
    st.markdown("After merging the different datasets into one master data, we processed it to obtain only the elementary school data, as well as created the different ratios we will be needing for the modelling. 14.73% or 5,282 rows were removed from the elementary school data when the outliers were filtered out.")
#-----End of Page 2 (Data Information)-----#


#-----Start of Page 3 (Methodology)-----#

elif my_page == 'Methodology':
    st.title("Methodology")
    st.image('./images/methodology.png',use_column_width=True)
    st.markdown("We created a master data where we merged all of the seven source data. We then proceeded to drop the zero and missing values, and filtered the dataset to elementary.")
    st.markdown("From there, we engineered three features that we deemed helpful in answering our problem (such as the student-teacher ratio, student per school room ratio, and MOOE per student ratio). We proceeded to prepare the data for the modelling by dropping the outliers so that they won’t skew the clustering. After dropping the outliers, the dataset is scaled and then fed to different models.")
    st.markdown("We tested both Hierarchical and KMeans modelling and settled on using KMeans with 3 clusters based on the model’s silhouette score and the amount of distribution among the clusters.")

#-----End of Page 3 (Methodology)-----#

#-----Start of Page 4 (EDA)-----#
elif my_page == 'EDA':
    st.title("Exploratory Data Analysis")
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between subheader and graph #
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between subheader and graph #
    st.subheader("Student-Teacher Ratio")
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
  
    # end of data prep for maps #
    
    # start of generating map for student/teacher ratio #
    c1, c2 = st.beta_columns(2)

    fig, ax = plt.subplots(1, figsize=(18, 10))

    cmap = mpl.cm.Oranges(np.linspace(-0.5,1,23))
    cmap = mcolors.ListedColormap(cmap[10:,:-1])

    merged_data.plot(column=var1, cmap=cmap, linewidth=0.8, ax=ax, edgecolor='0.8', vmin=vmin1, vmax=vmax1)
    ax.grid(False)
    plt.title("Average Student-Teacher Ratio Per Province",fontsize=18,y=1.02)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin1, vmax=vmax1))
    cbar = fig.colorbar(sm)

    # end of generating map for student/teacher ratio #
    
    c1.pyplot(fig) # show graph #
    
    # start of insights #
    c2.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between graph and insights #
    c2.markdown(':bulb: Provinces like Rizal, Bulacan, and Cebu have the highest average student-teacher ratio; while most of the regions in Luzon like CAR, Region I and II enjoys the lowest average student-teacher ratio.')
    # end of insights #
    
    # start of generating map for student per room ratio #
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between subheader and graph #
    st.subheader("Student Per Room Ratio")
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between subheader and graph #
    c3, c4 = st.beta_columns(2)

    fig, ax = plt.subplots(1, figsize=(18, 10))

    cmap = mpl.cm.Oranges(np.linspace(-0.5,1,23))
    cmap = mcolors.ListedColormap(cmap[10:,:-1])

    merged_data.plot(column=var2, cmap=cmap, linewidth=0.8, ax=ax, edgecolor='0.8', vmin=vmin2, vmax=vmax2)
    ax.grid(False)
    plt.title("Average Student Per Room Ratio Per Province",fontsize=18,y=1.02)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin2, vmax=vmax2))
    cbar = fig.colorbar(sm)

    # end of generating map for student per room ratio #
    
    c3.pyplot(fig) # show graph #
    
    # start of insights #
    c4.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between graph and insights #
    c4.markdown(':bulb: High student-room ratio in NCR, South Luzon, and Mindanao; lower student-room ratio in North Luzon.')
    # end of insights #
    
    # start of generating map for mooe per student ratio #
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between subheader and graph #
    st.subheader("MOOE Per Student Ratio")
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between subheader and graph #
    c5, c6 = st.beta_columns(2)

    fig, ax = plt.subplots(1, figsize=(18, 10))

    cmap = mpl.cm.Oranges(np.linspace(-0.5,1,23))
    cmap = mcolors.ListedColormap(cmap[10:,:-1])

    merged_data.plot(column=var3, cmap=cmap, linewidth=0.8, ax=ax, edgecolor='0.8', vmin=vmin3, vmax=vmax3)
    ax.grid(False)
    plt.title("Average Student Per Room Ratio Per Province",fontsize=18,y=1.02)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin3, vmax=vmax3))
    cbar = fig.colorbar(sm)

    # end of generating map for mooe per student ratio #
    
    c5.pyplot(fig) # show graph #
    
    # start of insights #
    c6.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between graph and insights #
    c6.markdown(':bulb: High MOOE per student in Northern Luzon, Samar, Zamboanga; low MOOE per student in Southern Luzon, and the regions around Cotabato.')
    # end of insights #
        
#-----End of Page 4 (EDA)-----#

#-----Start of Page 5 (Cluster Analysis)-----#

elif my_page == 'Cluster Analysis':
    
    st.title("Cluster Analysis")
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space subheader and graph #
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space subheader and graph #
    
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
    from math import pi
    
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
    st.markdown('<div style="font-size: medium;padding-bottom: 15px;font-weight:bold">Cluster 1: Schools with low demand & high resources</div>',unsafe_allow_html=True)
    st.markdown('• Mostly situated in rural areas')
    st.markdown('• Low student-teacher ratio')
    st.markdown('• Low student per room ratio')
    st.markdown('• High MOOE per student')
    
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between graph and insights #
    st.markdown('<div style="font-size: medium;padding-bottom: 15px;font-weight:bold">Cluster 2: Schools with high demand & low resources</div>',unsafe_allow_html=True)
    st.markdown('• Mostly situated in urban areas')
    st.markdown('• High student-teacher ratio')
    st.markdown('• High student per room ratio')
    st.markdown('• Low MOOE per student')
    
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between graph and insights #
    st.markdown('<div style="font-size: medium;padding-bottom: 15px;font-weight:bold">Cluster 3: Schools with moderate demand & resources</div>',unsafe_allow_html=True)
    st.markdown('• Found in both urban/rural areas')
    st.markdown('• Moderate student-teacher ratio')
    st.markdown('• Moderate student per room ratio')
    st.markdown('• Moderate MOOE per student')
    # end of insights #
    
    #---- Mapping of Clusters----
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True)
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True)
    st.subheader("Mapping schools belonging to the different clusters")
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True)
    
    from shapely.geometry import Point, Polygon
    
    geometry = [Point(xy) for xy in zip(df_map['Longitude'], df_map['Latitude'])]
    geo_df = gpd.GeoDataFrame(df_map, geometry = geometry)
    
    c1, c2 = st.beta_columns(2)
    option = c2.selectbox('Which cluster do you want to see?', [1, 2, 3])
    
    if option == 1:
        color = "#A6C3D0"
        c2.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True)
        c2.markdown(':bulb: Cluster 1 is abundant in upper Luzon (e.g. Isabela, Iloilo, Pangasinan), Western & Eastern Visayas (e.g. Leyte, Bohol). It is least common in NCR.')
    elif option == 2:
        color = "#BD4C2F"
        c2.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True)
        c2.markdown(':bulb: Cluster 2 is abundant in Central to South Luzon (e.g. CALABARZON, Bulacan), Central Visayas (e.g. Cebu). It is most common in NCR')
    else:
        color = "#FccF55"
        c2.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True)
        c2.markdown(':bulb: Cluster 3 is abundant in upper Luzon (e.g. Pangasinan, Iloilo, Isabela), Western & Eastern Visayas (e.g. Leyte, Negros Occidental).')
        
    fig, ax = plt.subplots(figsize=(15, 15))
    shapefile.plot(ax=ax, alpha = 0.4, color='grey')
    geo_df[geo_df['cluster']==(option-1)].plot(ax=ax, marker='+',color=color, markersize=8)
    plt.title("Map of Schools in Cluster " + str(option) ,fontsize=22,y=1.02)
    
    # end of generating map #
    
    c1.pyplot(fig) # show graph #
    
#-----End of Page 5 (Cluster Analysis)-----#

#-----Start of Page 6 (Other Cluster Insights)-----#

elif my_page == 'Other Cluster Insights':
    
    st.subheader("Majority of Elementary Schools in the Philippines are managed by DepEd and are in a monograde setup")
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True)
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True)

    # School Management Clusters
    df_cluster_feat = df_all[["student_per_rm","mooe_student_ratio","student_teacher_ratio", "school.classification2", 
                            "school.organization", "cluster"]]
    df_class2 = df_cluster_feat.groupby('cluster')['school.classification2'].value_counts().to_frame()
    df_class2 = df_class2.rename(columns={'school.classification2' : 'total_count'})
    df_class2 = df_class2.reset_index()
    
    mycolors = ["#A6C3D0","#BD4C2F","#FccF55"]
    sns.set_palette(sns.color_palette(mycolors))
    sns.set_style("whitegrid")
    fig=plt.figure(figsize=(9,7))
    sns.barplot(x='cluster', y='total_count', hue='school.classification2', data=df_class2)
    plt.xlabel('Clusters',fontsize=12)
    plt.ylabel("Number of Schools",fontsize=12)
    plt.xticks(ticks = (0, 1, 2), labels = ['1', '2', '3'])
    plt.title("School Management Classification per Cluster",fontsize=15,y=1.02)
    st.pyplot(fig) # show graph #
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True)
    
    # School Organization Clusters
    df_org = df_cluster_feat.groupby('cluster')['school.organization'].value_counts().to_frame()
    df_org = df_org.rename(columns={'school.organization' : 'count'})
    df_org = df_org.reset_index()
    
    mycolors = ["#A6C3D0","#BD4C2F","#FccF55"]
    sns.set_palette(sns.color_palette(mycolors))
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(9,7))
    sns.barplot(x='cluster', y='count', hue='school.organization', data=df_org)
    plt.xlabel('Clusters',fontsize=12)
    plt.ylabel("Number of Schools",fontsize=12)
    plt.xticks(ticks = (0, 1, 2), labels = ['1', '2', '3'])
    plt.title("School Organization per Cluster",fontsize=15,y=1.02)
    st.pyplot(fig)
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True)
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True)
    
    # SPED Clusters
    st.subheader("SPED opportunities are present in clusters where schools are mostly in urban locations.")
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True)
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True)
    
    spedcount=df_all.groupby("cluster")["total_enrollees_sped"].sum().to_frame().reset_index()
    spedcount.rename(columns={ spedcount.columns[0]: "Cluster" }, inplace = True)
    spedcount.rename(columns={ spedcount.columns[1]: "No. of SPED Enrollees" }, inplace = True)
    
    fig=plt.figure(figsize=(9,7))
    sns.barplot(x='Cluster', y='No. of SPED Enrollees', data=spedcount)
    plt.xlabel('Clusters',fontsize=12)
    plt.ylabel("SPED Enrollees",fontsize=12)
    plt.xticks(ticks = (0, 1, 2), labels = ['1', '2', '3'])
    plt.title("Incidence of SPED Enrollees per Cluster",fontsize=15,y=1.02)
    st.pyplot(fig) # show graph #
    
    # end of insights #
    
#-----End of Page 6 (Other Cluster Insights)-----#


#-----Start of Page 7 (Conclusions and Recommendations)-----#

elif my_page == 'Conclusions and Recommendations':

    st.title("Conclusions and Recommendations")
    st.image('./images/conclusion.jpg',use_column_width=True)
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between subheader and paragraph #
    st.subheader("Regarding the Philippine Education Based on the Data Insights")
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between title and subheader #
    st.markdown(':bulb:  Consider giving more resources to urban areas given their higher school congestion')
    st.markdown(':bulb:  Examine reasons for low congestion in rural areas')
    st.markdown('  - Look into other factors that affect access to schools (e.g. transportation, economic situation of families, etc) that may prevent students from going to schools - that may lead to underutilization of  resources')
    st.markdown(':bulb:  Address the low incedence of SPED enrollees in rural areas')
    st.markdown('  - For areas which do not have SPED offerings, where do students who need SPED go to?')
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between title and subheader #
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between title and subheader #
    st.subheader("Regarding the Machine Learning Aspect")
    st.markdown('<div style="color: #FFFFFF;">.</div>',unsafe_allow_html=True) # for space between title and subheader #
    st.markdown(':bulb:  Look into other features which may better categorize schools and improve clustering statistics like the inertia score and silhouette score.')
    st.markdown(':bulb:  Experiment with the different hyperparameters for K-Means and Hierarchical Clustering (init, distance metrics, linkages). Also check if there are other algorithms that may result to better clustering.')
#-----End of Page 7 (Conclusion and Recommendation)-----#