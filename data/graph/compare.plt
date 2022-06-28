set term png
set output "compare.png"
set key out top center horizontal
set style data linespoints
set style line 1 lc rgb "dark-red" pt 7 ps 0.7 lt 1 lw 2
set style line 2 lc rgb "dark-green" pt 7 ps 0.7 lt 1 lw 2
set yrange [0:]
set grid ytics
set xlabel "Load factor (%)" font ",16"
set ylabel "Skewness ({/Symbol q})" font ",16"
plot "vqf" using 1:2 ls 1 title "Vector Quotient Filter", \
       "cvqf" using 1:2 ls 2 title "Counting Vector Quotient Filter"
