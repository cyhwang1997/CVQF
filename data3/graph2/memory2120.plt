set term eps
set lmargin 14
set rmargin 4
set bmargin 5
set output "memory2120.eps"
set key out top center horizontal font ",30"
set style data linespoints
set style line 1 lc rgb "dark-red" pt 7 ps 0.7 lt 1 lw 2
set style line 2 lc rgb "dark-green" pt 7 ps 0.7 lt 1 lw 2
set grid ytics
set yrange [0:]
set xrange [:0.9]
set ytic 200 font ",30"
set xtic 0.2 font ",30"
set xlabel "Skewness ({/Symbol q})" offset 0,-1 font ",30"
set ylabel "Memory (kB)" offset -4 font ",30"
plot "memory2120" using 1:2 ls 1 title "CVQF"
