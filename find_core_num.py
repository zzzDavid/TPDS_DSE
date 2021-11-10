rsc_table = {'244': dict(), '488' : dict(), '888' : dict()}
rsc_table['244'] = {'lut' : 11.339, 'reg' : 12.664, 'bram' : 5, 'uram' : 13, 'dsp' : 24}
rsc_table['488'] = {'lut' : 20.538, 'reg' : 25.453, 'bram' : 5, 'uram' : 23, 'dsp' : 160}
rsc_table['888'] = {'lut' : 31.956, 'reg' : 38.980, 'bram' : 5, 'uram' : 35, 'dsp' : 320}
rsc_table['EthernetIP'] = {'lut' : 367.9, 'reg' : 314.7, 'bram' : 903, 'uram': 0, 'dsp' : 0}
rsc_table['HyperConnect'] = {'lut' : 12.1, 'reg' : 5.2, 'bram' : 0, 'uram': 0, 'dsp' : 0}

u200 = {'lut' : 870, 'reg' : 723+331+723, 'bram' : 638+326+638, 'uram': 320+160+320, 'dsp' : 2265+1317+2265, 'name' : 'u200'} 
u280 = {'lut' : 1304, 'reg' : 2607, 'bram' : 2016, 'uram': 960, 'dsp' : 9024, 'name': 'u280'}

def calc_core_num(device, core):
    core_num = dict()
    core_num_list = list()
    rcs_types = ['lut', 'reg', 'sram', 'dsp']
    for rcs in rcs_types:
        if rcs == 'sram':
            available = device['bram'] * 36 + device['uram'] * 288
            overhead = rsc_table['EthernetIP']['bram'] * 36 + rsc_table['EthernetIP']['uram'] * 288
            unit = rsc_table[core]['bram'] * 36 + rsc_table[core]['uram'] * 288
        else:
            available = device[rcs]
            overhead = rsc_table['EthernetIP'][rcs] + rsc_table['HyperConnect'][rcs]
            unit = rsc_table[core][rcs]
        core_num[rcs] = (available - overhead) / unit
        core_num_list.append(core_num[rcs])
    return core_num, int(int(min(core_num_list)) / 4)

if __name__ == "__main__":
    for device in [u200, u280]: 
        for core in ['244', '488', '888']:
            dev = device['name']
            core_num, rec = calc_core_num(device, core)
            print(f"Number of {core} cores that can fit in device {dev}: ")
            print(core_num)
            print(f"recommended core per group: {rec}\n\n")
