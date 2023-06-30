## Spectrophotometer Correction Tools
1. [x] Spectrum data (.spd) must be exported eighter inside of Measurement device software or using proper decoder.
2. [x] Default format is tsv  and header "nm   Abs" in this scenario all floating numbers delimiting with comma(","")
3. [x] New line character is "\n"
4. [x] Seperator character is "\t"


``` bash
TEST_DATA = 'data/neu_cal/Test 2/Neu-2_5x10e-2-I.txt'
TEST_BASE = 'data/neu_cal/Test 2/Neu-2_5x10e-2-base.txt'
GLOB_BASE = 'data/neu_cal/Test 12/Neu-3x10e-4-base-III.txt'
```

For autonomous data convolution specified spectrophotometric absorbance data
Filenames and baseline names had to be added accordingly
Filename and baseline names must be the same excluding the flag word
For example flag is - base there could be more data numerated accordingly they all will be corrected automaticly
according the TEST_BASE data also there could be another base file for same measurement then this bases should be 
numarated accordingly. Algorithm convolve bases inside of given data.

* Number of data = 3
* Number of base = 2
* => Output 6 baselined output

``` bash
* TEST_DATA = 'Neu-2_5x10e-2-I.txt'
* TEST_BASE = 'Neu-2_5x10e-2-base.txt'
```
###
