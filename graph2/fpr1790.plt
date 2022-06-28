set term eps
set output "fpr1790.eps"
set key out top center horizontal font ",16"
set style data linespoints
set style line 1 lc rgb "dark-blue" pt 7 ps 0.7 lt 1 lw 2
set grid ytics
set xlabel "Skewness ({/Symbol q})" font ",16"
set ylabel "1/{/Symbol e}" font ",16"
plot "fpr1790" using 1:2 ls 1 title "CVQF"
