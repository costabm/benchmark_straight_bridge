import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# with open("other/polimi_raw_data/postprocess_data.txt", "r") as file:
#     raw_lst_str = file.readlines()  # raw list of strings

with open("other/polimi_raw_data/postprocess_data_(BC_edited).txt", "r") as file:
    raw_lst_str = file.readlines()  # raw list of strings


d = {}
test_list = []

def get_substring_between(str, start, end):
    return str[str.find(start) + len(start) : str.rfind(end)]

for l, line in enumerate(raw_lst_str):
    if line == "Inizio postprocess\n":
        d = {}

        d['Temperature [C]'] = float(get_substring_between(raw_lst_str[l + 2], "Temperatura [°C]:", "\n"))
        d["Pressure [Pa]"] = float(get_substring_between(raw_lst_str[l + 3], "Pressione assoluta [Pa]:", "\n"))
        d["Density [kg/m^3]"] = float(get_substring_between(raw_lst_str[l + 4], "'rho [kg/m^3]:", "\n"))
        d["Pitot V Galleria"] = float(get_substring_between(raw_lst_str[l + 5], "Pitot V Galleria", "V Campo"))
        d["V Campo"] = float(get_substring_between(raw_lst_str[l + 5], "V Campo", "VCampo/VGalleria"))
        d["VCampo/VGalleria"] = float(get_substring_between(raw_lst_str[l + 5], "VCampo/VGalleria", "\n"))

        d["Sensor(1)"] = get_substring_between(raw_lst_str[l + 6], "Bilancia:	", "\n")
        d["Forces(1)"] = np.array(raw_lst_str[l + 8].split('\t')[:-1], dtype=float)
        d["Cf(1)"] = np.array(raw_lst_str[l + 10].split("\t")[:-1], dtype=float)

        d["Sensor(2)"] = get_substring_between(raw_lst_str[l + 12], "Bilancia:	", "\n")
        d["Forces(2)"] = np.array(raw_lst_str[l + 14].split('\t')[:-1], dtype=float)
        d["Cf(2)"] = np.array(raw_lst_str[l + 16].split("\t")[:-1], dtype=float)

        d["Sensor(3)"] = get_substring_between(raw_lst_str[l + 18], "Bilancia:	", "\n")
        d["Forces(3)"] = np.array(raw_lst_str[l + 20].split('\t')[:-1], dtype=float)
        d["Cf(3)"] = np.array(raw_lst_str[l + 22].split("\t")[:-1], dtype=float)

        d["Sensor(4)"] = get_substring_between(raw_lst_str[l + 24], "Bilancia:	", "\n")
        d["Forces(4)"] = np.array(raw_lst_str[l + 26].split('\t')[:-1], dtype=float)
        d["Cf(4)"] = np.array(raw_lst_str[l + 28].split("\t")[:-1], dtype=float)

        d["Message"] = raw_lst_str[l + 31][:-1]
        d["Tag"] = get_substring_between(raw_lst_str[l + 33], "\ATI", r"\n")[1:]
        d["Case"] = d["Tag"][:3]
        d["P"] = get_substring_between(raw_lst_str[l + 33], "ATI_", r"_")
        d["Yaw"] = float(get_substring_between(raw_lst_str[l + 33], "Ang", r"\n"))
        test_list.append(d)

df = pd.DataFrame(test_list)
df_1 = df[df['P']=='P7000']


plt.scatter(df_1['Yaw'],np.stack(df_1['Forces(4)'])[:, 5])
plt.grid()
plt.savefig(r'other\polimi_raw_data\pontoon_RZ.jpg')
plt.close()

