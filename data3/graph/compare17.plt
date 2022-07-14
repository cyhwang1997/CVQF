set term eps
set lmargin 15
set rmargin 6
set bmargin 5
set output "compare17.eps"
set key out top center horizontal font ",30"
set style data linespoints
set style line 1 lc rgb "dark-red" pt 7 ps 0.7 lt 1 lw 2
set style line 2 lc rgb "dark-green" pt 7 ps 0.7 lt 1 lw 2
set yrange [0:]
set ytics 0.2 font ",30"
set xtics 20 font ",30"
set grid ytics
set xlabel "Load factor (%)" offset 0,-1 font ",30"
set ylabel "Skewness ({/Symbol q})" offset -5 font ",30"
plot "vqf17" using 1:2 ls 1 title "VQF", \
       "cvqf17" using 1:2 ls 2 title "CVQF"
