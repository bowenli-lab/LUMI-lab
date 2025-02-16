import matplotlib.pyplot as plt
import numpy as np
#
# data = {'A': [368, 35484, 9213, 2019, 36, 45546, 392, 713, 1028, 335, 420, 185], 'B': [496, 1018, 198, 1759, 288, 1549, 177, 167, 1348, 177, 198, 321], 'C': [16, 1865, 936, 296, 91848, 339, 1031, 146, 262, 84, 217, 381], 'D': [50691, 272, 310, 158, 373, 2521, 636, 49, 442, 86, 633, 640], 'E': [420, 7976, 309, 620, 452, 804, 1029, 294, 1259, 1789, 800, 766], 'F': [168, 4475, 238, 98, 1290, 907, 213, 560, 73, 222, 604, 192], 'G': [190, 320, 1763, 458, 912, 274, 294, 710, 192, 132, 1155, 589], 'H': [12, 5688, 798, 1477, 1402, 395, 152, 243, 406, 130, 229, 242]}
#
#
# # take log of the data
# # for key in data.keys():
# #     data[key] = [np.log10(x) for x in data[key]]
# matrix = np.array([data[row] for row in data])
#
# plt.figure(figsize=(12, 8))
# plt.imshow(matrix, cmap='viridis', aspect='auto')
# plt.colorbar(label='Readout Value')
#
# plt.xticks(ticks=np.arange(12), labels=[str(i) for i in range(1, 13)])
# plt.yticks(ticks=np.arange(8), labels=list(data.keys()))
#
# plt.title('96-Well Plate Readout Heatmap')
# plt.xlabel('Column')
# plt.ylabel('Row')
#
# for i in range(matrix.shape[0]):
#     for j in range(matrix.shape[1]):
#         plt.text(j, i, f"{list(data.keys())[i]}{j+1}", ha='center', va='center', color='white')
#
# plt.savefig('96_well_plate_heatmap.jpg')
# plt.show()




import plotly.graph_objects as go
import numpy as np
import requests

base_url = "http://localhost:8000"

entry_id = '667abe9363d44fdd873d8050'
request_url = base_url + f"/entry/{entry_id}/readings"

response = requests.get(request_url)
data = response.json()



# Extracting readings and hover text
wells = sorted(data['0'].keys())
rows = sorted(list(set(well[0] for well in wells)), reverse=True)
columns = sorted(list(set(int(well[1:]) for well in wells)))

heatmap_data = np.zeros((len(rows), len(columns)))
hovertext = np.empty((len(rows), len(columns)), dtype=object)

for well in wells:
    row_idx = rows.index(well[0])
    col_idx = columns.index(int(well[1:]))
    heatmap_data[row_idx, col_idx] = data['0'][well]['reading']

    components = data['0'][well]['components']
    hovertext[row_idx, col_idx] = (f"Well: {well}<br>Reading: {data['0'][well]['reading']}"
                                   f"<br>Amines: {components['amines']}<br>Isocyanide: {components['isocyanide']}"
                                   f"<br>Lipid Carboxylic Acid: {components['lipid_carboxylic_acid']}"
                                   f"<br>Lipid Aldehyde: {components['lipid_aldehyde']}")

fig = go.Figure(data=go.Heatmap(
    z=heatmap_data,
    x=[str(col) for col in columns],
    y=rows,
    text=hovertext,
    hoverinfo="text",
    colorscale='Viridis'
))

fig.update_layout(
    title='96-well Plate Readings Heatmap',
    xaxis_title='Column',
    yaxis_title='Row'
)

fig.show()
