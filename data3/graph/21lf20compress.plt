set term eps
set lmargin 13
set rmargin 4
set bmargin 5
set output "21lf20compress.eps"
set key out top center horizontal font ",30"
set style data linespoints
set style line 1 lc rgb "dark-blue" pt 7 ps 0.7 lt 1 lw 2
set yrange [0:]
set xrange [:0.9]
set ytic 20 font ",30"
set xtic 0.2 font ",30"
set grid ytics
set xlabel "Skewness ({/Symbol q})" offset 0,-1 font ",30"
set ylabel "SLOTS SAVED (%)" offset -3 font ",30"
plot "21lf20compress" using 1:2 ls 1 title "CVQF"
