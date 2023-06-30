import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def combine_filename_base(file: str, base: str, base_specifier: str = '-base') -> str:
    # Combines filenames with their bases
    # measurements and bases has to be in same structure
    # base specifier could be changed from its default value which is 'base'
    # base specifier has to be and of naming structure
    # {NAME} {ITERATOR} , {BASE_SPECIFIER} {BASE_ITERATOR} has to be seperated with dashes(-)
    # Ex: measurement name = {NAME}-{MEASUREMENT_ITERATOR}.txt
    # Ex: base name = {NAME}-base-{BASE_ITERATOR}.txt
    path = file.split('/')[:-1]
    filename = file.split('/')[-1].split('.')[:-1]
    baseStr = base[base.find(base_specifier):]
    base_name = baseStr.split('.')[:-1]

    # print(path, filename, baseStr, base_name)
    filename = filename[0] + '_cor_v_' + base_name[0]

    return f'{"/".join(path + [filename])}.txt'


def test_base_corrections(folder, g_base):
    for g in glob.glob(f'{folder}/*'):
        base = glob.glob(str(g) + '/*base*.txt')
        for b in base:
            # print(b)
            base_data = load_from_txt(b)
            peak_val = peak_calculate(base_data)
            data_corrected = vectorel_baseline_correction(base_data, base_data, g_base)
            plot_comparesion(base_data, data_corrected)
            # plot_spectrum(base_data, title=f"{b.split('/')[-1]} {peak_val} ")
            # plot_spectrum(data_corrected, title=f"Normalized {b.split('/')[-1]} {peak_calculate(data_corrected)}")


def bulk_data_correction(folder: str, g_base,prefix="base", save=False, plot=False):
    base_prefix = f"/*{prefix}*.txt"
    for g in glob.glob(f'{folder}/*'):
        base = glob.glob(str(g) + base_prefix)
        for b in base:
            # print(b)
            base_data = load_from_txt(b)
            # print(f"result={peak_calculate(base_data)}")
            peakVal = peak_calculate(base_data)
            plot_spectrum(base_data, title=f"{b.split('/')[-1]} {peakVal} ")
            for file in set(glob.glob(str(g) + '/*.txt')) - set(base):
                # print(b)
                baseStr = b[0:b.find(f'-{prefix}')]
                isThisMyBase = file.startswith(baseStr)
                # print(file,baseStr,isThisMyBase)
                if isThisMyBase:
                    data = load_from_txt(file)
                    # data_corrected = scalar_baseline_correction(data, peakVal, 0.45) -covid 0.346
                    data_corrected = vectorel_baseline_correction(data, base_data, g_base)
                    if plot:
                        plot_spectrum(data, title=f"{file.split('/')[-1]} {peak_calculate(data)}")
                        plot_spectrum(data_corrected,
                                      title=f"Normalized {file.split('/')[-1]} {peak_calculate(data_corrected)}")
                        plot_comparesion(base_data, data_corrected)
                    # print(f"result={peak_calculate(data)}")
                    # print(f"normalized={peak_calculate(data_corrected)}")
                    if save:
                        save_spectrum_to_txt(combine_filename_base(file, b,base_specifier=f"-{prefix}"), data_corrected)


def load_from_txt(file) -> np.object:
    data = np.loadtxt(file)
    return data


def peak_calculate(data) -> float:
    start_nm = 500
    end_nm = 650
    range_indexes = np.where((data > start_nm) & (data < end_nm))
    max_ranged_val = np.max(data[range_indexes, 1])
    # print(data[np.argwhere(data == max_ranged_val)])
    # print(data)
    return max_ranged_val


def scalar_baseline_correction(data, peak_val, base):
    data = data.copy()
    multipleer = base / peak_val
    print(multipleer, base, peak_val)
    data[:, 1] = data[:, 1] * multipleer
    return data


def vectorel_baseline_correction_with_peak(data, peak_val, base_val):
    data = data.copy()
    multipleer = base_val[:, 1] / peak_val
    # print(multipleer, baseVal, peakVal)
    data[:, 1] = data[:, 1] * multipleer
    return data


def vectorel_baseline_correction(data, peak_val, base_val):
    data = data.copy()
    multipleer = base_val[:, 1] / peak_val[:, 1]
    # print(multipleer, baseVal, peakVal)
    data[:, 1] = data[:, 1] * multipleer
    return data


def plot_spectrum(data, title=""):
    f = plt.figure()
    f.set_figwidth(20)
    f.set_figheight(8)
    plt.scatter(data[:, 0], data[:, 1])
    plt.title(title)
    # plt.figure(figsize=(120, 60))
    plt.show()


def plot_comparesion(base, result):
    f = plt.figure()
    f.set_figwidth(20)
    f.set_figheight(8)
    plt.scatter(base[:, 0], base[:, 1], color='b', label='base')
    plt.scatter(result[:, 0], result[:, 1], color='r', label='result')
    difference = result[:, 1] - base[:, 1]
    plt.scatter(result[:, 0], difference, color='g', label='compare')
    plt.plot([np.argmax(base[200:300, 1]) + 500,
              np.argmax(base[200:300, 1]) + 500,
              ], [0, 0.5])
    # plt.plot([np.argmax(data2[200:300, 1]) + 500, np.argmax(data2[200:300, 1]) + 500], [0, 0.5])
    # plt.plot([np.argmax(difference[200:300]) + 500, np.max(difference[200:300]) + 500], [0, 0.5])
    # plt.plot([300, 800], [np.max(data[200:300, 1]), np.max(data[200:300, 1])])
    plt.title(f'x:{np.argmax(base[:, 1])} y:{np.max(base[:, 1])} ')
    # plt.figure(figsize=(120, 60))
    plt.legend()
    plt.show()


def save_spectrum_to_normal_tsv(file_name, data):
    os.makedirs(f'data/{os.path.dirname(file_name)}', exist_ok=True)
    np.savetxt(f'data/{file_name}', data, fmt='%.2f %.3f', delimiter='\t', header="nm Abs")


def save_spectrum_to_txt(file_name, data):
    df = pd.DataFrame({"nm": data[:, 0], "Abs": data[:, 1]})
    os.makedirs(f'tdata/{os.path.dirname(file_name)}', exist_ok=True)
    df.to_csv(f'tdata/{file_name}', index=False, sep='\t', decimal=',')


def test():
    baseVal = 0.45
    peakVal = peak_calculate(load_from_txt(TEST_BASE))
    data = load_from_txt(TEST_DATA)
    data_corrected = scalar_baseline_correction(data, peakVal, baseVal)
    base_data = load_from_txt(GLOB_BASE)
    data_corrected_vectorel_peak = vectorel_baseline_correction_with_peak(data, peakVal, base_data)
    data_corrected_vectorel = vectorel_baseline_correction(data,
                                                           load_from_txt('data/neu_cal/Test 2/Neu-2_5x10e-2-base.txt'),
                                                           base_data)
    plot_spectrum(data, title="orj")
    plot_spectrum(data_corrected, title="corrected")
    plot_spectrum(data_corrected_vectorel_peak, title="corrected_vectorel_peak_val")
    plot_spectrum(data_corrected_vectorel, title="corrected_vectorel")
    print(f"result={peak_calculate(data)}")
    print(f"normalized={peak_calculate(data_corrected)}")
    print(f"normalized_vectorel={peak_calculate(data_corrected_vectorel)}")
    save_spectrum_to_txt(TEST_DATA, data_corrected)
    combine_filename_base(TEST_DATA, TEST_BASE)


TEST_DATA = 'data/neu_cal/Test 2/Neu-2_5x10e-2-I.txt'
TEST_BASE = 'data/neu_cal/Test 2/Neu-2_5x10e-2-base.txt'
GLOB_BASE = 'data/inf/neu_cal/Test 12/Neu-3x10e-4-base-III.txt'
# COVID_BASE = 'data/covid/Örnek Deneme/300ng-base-III.txt'
COVID_BASE = 'data/covid/AuNP-II.txt' #peak_pos:522nm peak_abs:0.495

if __name__ == '__main__':
    # test()
    base_data = load_from_txt(COVID_BASE)

    plot_spectrum(base_data, f'base {peak_calculate(base_data)}')
    bulk_data_correction('data/covid/Kalibrasyon', base_data,'base', save=True, plot=True)
    bulk_data_correction('data/covid/Kalibrasyon', base_data,'Base', save=True, plot=True)
    # folder = 'data/covid/Kalibrasyon/Test 5'
    # for g in glob.glob(f'{folder}/*.txt'):
    #     plot_spectrum(load_from_txt(g), g.split('/')[-1])
    # base_corrections('data/neu_cal', base_data)
    # print_hi('PyCharm')
