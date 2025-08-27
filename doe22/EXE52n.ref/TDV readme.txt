The Time-Dependent Valuation (TDV) files contain hourly multiplier values for
electric, natural gas and propane use which convert the hourly energy demands
into hourly TDV energy. There are TDV multiplier values for residential and
non-residential buildings.

DOE-2 versions reads a single file named TDVCTZ.bin. The default distributed
content of this file is the non-residential version for the current in effect
Title 24. (so the April 2016 release uses 2013 non-res - TDVCTZnr_2013.bin -
as the default file named TDVCTZ.bin)

The distribution contains two files (the residential - r - and non-residential
- nr - version) for each of the Title 24 vintages (2006, 2008, 2013 and 2016 and
2019).

If you desire to switch to use a different vintage or a different sector, just
copy the desired file over the default TDVCTZ.bin file. For example, to use the
2016 residential TDV's copy TDVCTZr_2016.bin onto TDVCTZ.bin.