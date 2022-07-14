set term eps
set lmargin 17
set rmargin 4
set bmargin 5
set output "fpr2120.eps"
set key out top center horizontal font ",30"
set style data linespoints
set style line 1 lc rgb "dark-blue" pt 7 ps 0.7 lt 1 lw 2
set grid ytics
set yrange [:3500]
set xrange [:0.9]
set ytic 1000 font ",30"
set xtic 0.2 font ",30"
set xlabel "Skewness ({/Symbol q})" offset 0,-1 font ",30"
set ylabel "1/{/Symbol e}" offset -6 font ",30"
plot "fpr2120" using 1:2 ls 1 title "CVQF"
