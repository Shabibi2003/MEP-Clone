echo OFF
set doedir=c:\doe22
set weather.bin=%doedir%\weather\%1.bin 
set WEATHER.FMT=%doedir%\weather\%1.ft
bin2txt
echo ON
