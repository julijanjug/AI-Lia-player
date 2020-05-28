import os
import time

i = 0
while i < 100:
    print("iteracija", i)
    os.system("./lia generate Nabiralec_Vojak_map_fighter Nabiralec_Vojak_map_fighter")
    i += 1