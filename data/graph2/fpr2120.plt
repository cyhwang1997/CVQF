set term png
set output "fpr2120.png"
set key out top center horizontal
set style data linespoints
set style line 1 lc rgb "dark-blue" pt 7 ps 0.7 lt 1 lw 2
set grid ytics
set xlabel "Skewness ({/Symbol q})"
set ylabel "1 / False Positive Rate (1/{/Symbol e})"
plot "fpr2120" using 1:2 ls 1 title "Counting Vector Quotient Filter"
