set term eps
set output "memory2120.eps"
set key out top center horizontal font ",16"
set style data linespoints
set style line 1 lc rgb "dark-red" pt 7 ps 0.7 lt 1 lw 2
set style line 2 lc rgb "dark-green" pt 7 ps 0.7 lt 1 lw 2
set grid ytics
set yrange [0:]
set xlabel "Skewness ({/Symbol q})" font ",16"
set ylabel "Memory (kB)" font ",16"
plot "memory2120" using 1:2 ls 1 title "CVQF"
