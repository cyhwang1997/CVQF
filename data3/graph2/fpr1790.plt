set term eps
set lmargin 15
set rmargin 4
set bmargin 5
set output "fpr1790.eps"
set key out top center horizontal font ",30"
set style data linespoints
set style line 1 lc rgb "dark-blue" pt 7 ps 0.7 lt 1 lw 2
set grid ytics
set ytic 400 font ",30"
set xtic 0.2 font ",30"
set xlabel "Skewness ({/Symbol q})" offset 0,-1 font ",30"
set ylabel "1/{/Symbol e}" offset -4 font ",30"
plot "fpr1790" using 1:2 ls 1 title "CVQF"
