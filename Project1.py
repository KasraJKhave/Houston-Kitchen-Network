#!/usr/bin/env python
# coding: utf-8

# In[28]:

#The location source is the Loopnet.com website (https://www.loopnet.com/search/restaurants/for-lease/?sk=be2c51f63e8e6ed82bd98425d41bc40c&bb=k_h95lnolJy49p9y3D)
#We first need to put the Project1.py file and Houston_25.png
# and location_15.xlsx in the same folder. 


#import libraries


#!pip install matplotlib
# pip install pandas
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic as GD
from PIL import Image
import matplotlib.pyplot as plt
import random




#---------------------------task 1 , part 3---------------------------
#import the excel file and find the latitudes and longitudes
file_path = "Location_25.xlsx"
a = pd.read_excel(file_path, header=None)
print(a[0])
geolocator = Nominatim(user_agent="mehrsalami1998@gmail.com")
lat=[]
lon=[]
all_restaurants=[]
for i in range(25):
    location = geolocator.geocode(a[0][i],timeout=None)
    lat.append(location.latitude)
    lon.append(location.longitude)
    all_restaurants.append(location.address)
    
#making sure that the distance between each two restaurants is less than 0.5 miles
dist=[]
for i in range(25):
    for j in range(25):
        if(i!=j):
            dist.append(GD((lat[i],lon[i]),(lat[j],lon[j])).mi)
            if (GD((lat[i],lon[i]),(lat[j],lon[j])).mi<0.5):
                print("less than 0.5 mile",i,j)
        else:
            dist.append(0)
dist_matrix = np.array(dist).reshape((25, 25))




#---------------------------task 1 , part 4---------------------------
#Plot locations of kitchen onto the map
map_image = Image.open('HOUSTON_25.png')
maxlon, minlon, maxlat, minlat = -95.0845, -95.785, 30.05785, 29.49065
data_points = [(minlat, maxlon), (maxlat, minlon)]  
plt.imshow(map_image, extent=[minlon, maxlon, minlat, maxlat])
for i in range(len(lat)):
    plt.scatter(lon[i],lat[i], color='blue', marker='o',s=12)  # Adjust color and marker as needed
plt.show()

#finding the max and min coordinates so that we can find the 2.5 miles coordinates of the map
#the 2.5 values are computed using converesions of miles to latitude and longitude
BBox = ((min(lon),   max(lon),      min(lat), max(lat)))
print(BBox)
def argmax(lst):
    return lst.index(max(lst))
def argmin(lst):
    return lst.index(min(lst))
print(argmax(lon),argmax(lat),argmin(lon),argmin(lat))
maxlon=-95.0845
minlon=-95.785
maxlat=30.05785
minlat=29.49065

#making sure that the distances are 2.5 miles
print(GD((lat[16],lon[16]),(lat[16],maxlon)).mi)
print(GD((lat[16],lon[16]),(minlat,lon[16])).mi)
print(GD((lat[17],lon[17]),(maxlat,lon[17])).mi)
print(GD((lat[1],lon[1]),(lat[1],minlon)).mi)




#---------------------------task 1 , part 5---------------------------
#generating random locations as the service stations
random.seed(42)  
lat_range = (-95.785, -95.0845)
lon_range = (29.49065, 30.05785)
random_points = []
for _ in range(50):
    random_lat = random.uniform(lat_range[0], lat_range[1])
    random_lon = random.uniform(lon_range[0], lon_range[1])
    random_points.append((random_lat, random_lon))

#plot kitchens (in blue) and service locations (in red) on the map
map_image = Image.open('HOUSTON_25.png')
data_points = [(minlat, maxlon), (maxlat, minlon)]  
plt.imshow(map_image, extent=[minlon, maxlon, minlat, maxlat])
for i in range(len(lat)):
    plt.scatter(lon[i],lat[i], color='blue', marker='o',s=12)  
for i in range(len(random_points)):
    plt.scatter(random_points[i][0],random_points[i][1], color='red', marker='o',s=12)  
plt.savefig("Locations.jpeg")
plt.show()



#---------------------------task 1 , part 6---------------------------
#Construct a Python table of the collected location data
from tabulate import tabulate
geolocator = Nominatim(user_agent="mehrsalami1998@gmail.com")

zipcode1=[]
zipcode2=[]
address1=[]
address2=[]

for i in range(25):
    location = geolocator.reverse((lat[i], lon[i]), language="en",timeout=None)
    address1 .append( location.address)
    zipcode1 .append( location.raw.get("address", {}).get("postcode", None))
for i in range(50):
    location = geolocator.reverse((random_points[i][1], random_points[i][0]), language="en",timeout=None)
    address2 .append( location.address)
    zipcode2 .append( location.raw.get("address", {}).get("postcode", None))
    
table_data = []
for i in range(len(zipcode1)):
    index = i + 1 
    zipcode = zipcode1[i]
    address = address1[i]
    coordinates = (lat[i], lon[i])
    table_data.append([index, address, zipcode, coordinates])
for i in range(len(zipcode2)):
    index = i + 1 + len(zipcode1)
    zipcode = zipcode2[i]
    address = address2[i]
    coordinates = (random_points[i][1], random_points[i][0])
    table_data.append([index, address, zipcode, coordinates])
headers = ["Index", "Street Address", "Zip Code", "Coordinates"]


table = tabulate(table_data, headers=headers, tablefmt="simple")
# print(table)
np.savetxt("Locations.txt", np.array([table]), fmt='%s')



#---------------------------task 1 , part 7---------------------------
#Construct the d matrix 
d = list()
def distance(lat1,lon1,lat2,lon2):
    return (GD((lat1,lon1),(lat2,lon2)).mi)


for i in range(25):
    for j in range(50):
        if(i!=j):
            d.append(distance(lat[i],lon[i],random_points[j][1],random_points[j][0]))
        else:
            d.append(0)

d= np.array(d).reshape((25, 50))
distances_df = pd.DataFrame(d)

# distances_df
distances_df.to_csv("Distances.csv", header=False, index=False)



#---------------------------task 2-------------------------------------
import pulp as pl
from pulp import *
from scipy.sparse import csr_matrix, save_npz
import numpy as np


# Define model
model = LpProblem("Total distance",LpMinimize)
I = range(25)
J = range(50)
# Define variable
z = LpVariable.dicts('z', ((i,j) for i in I for j in J), cat='Binary')
#Define objective function
model += (lpSum([d[i,j]*z[i,j] for i in I for j in J]),"Total Distance Traveled")
# Define constraints
for j in J:
    model += lpSum(z[i,j] for i in I) == 1
for i in I:
    model += lpSum(z[i,j] for j in J) == 2

model.solve()

for i, j in z:
    if (z[i, j].value()==1):
        print(f"{z[i, j].name} = {z[i, j].value()}")

optimal_objective_value = value(model.objective)
print(f"Optimal objective function value: {optimal_objective_value}")
model.writeMPS("AP.mps")

mat=np.zeros((25,50))
for i in range(25):
    for j in range(50):
        if z[(i,j)].varValue==1:
            mat[i][j]=1

sparse_matrix = csr_matrix(mat)
values = sparse_matrix.data
column_indices = sparse_matrix.indices
row_pointers = sparse_matrix.indptr

save_npz("Solution.csr", sparse_matrix)





#---------------------------task 3 , part 1---------------------------
#Build the OD table
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

od_table = []
for i, j in z:
    if z[i, j].value() == 1:
        origin_index = i + 1  
        destination_index = j + 1 
        distance = d[i, j]
        od_table.append([origin_index, destination_index, distance])

headers_od = ["Cloud kitchen index", "Service station index", "Distance"]
tabulated_result=tabulate(od_table, headers=headers_od, tablefmt="simple")
# print(tabulated_result)
np.savetxt("OD.txt", np.array([tabulated_result]), fmt='%s')



#---------------------------task 3 , part 1---------------------------
#Plot the solution onto the map
map_image = Image.open('HOUSTON_25.png')

fig, ax = plt.subplots()
ax.imshow(map_image, extent=[minlon, maxlon, minlat, maxlat])

for i in range(len(lat)):
    ax.scatter(lon[i], lat[i], color='blue', marker='o', s=12)  
for i in range(len(random_points)):
    ax.scatter(random_points[i][0], random_points[i][1], color='red', marker='o', s=12) 

for i, j in z:
    if z[i, j].value() == 1:
      # print(i,j)
        lat_point1, lon_point1 = lat[i], lon[i]
        lat_point2, lon_point2 = random_points[j][1], random_points[j][0]
        ax.plot([lon_point1, lon_point2], [lat_point1, lat_point2], color='black', linestyle='-',linewidth=0.5)
plt.savefig("Solution.jpeg")
plt.show()




#---------------------------task 3 , part 1---------------------------
#Create the frequency graph
short_range = medium_range = long_range = 0

for i, j, distance in od_table:
    if distance < 3:
        short_range += 1
    elif 3 <= distance <= 6:
        medium_range += 1
    else:
        long_range += 1

distance_ranges = ["< 3 miles", "3-6 miles", "> 6 miles"]
frequency_values = [short_range, medium_range, long_range]

plt.bar(distance_ranges, frequency_values, color=['green', 'orange', 'red'])
plt.xlabel("Distance range")
plt.ylabel("Frequency")
plt.title("Frequency of different distance ranges")

plt.savefig("Frequency.jpeg")
plt.show()


# In[ ]:




