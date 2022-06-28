set term png
set output "fpr17.png"
set key out top center horizontal
set style data linespoints
set style line 1 lc rgb "dark-red" pt 7 ps 0.7 lt 1 lw 2
set style line 2 lc rgb "dark-green" pt 7 ps 0.7 lt 1 lw 2
set logscale xy 2
set grid ytics
set xlabel "log_2(100{/Symbol a})"
set ylabel "-log_2({/Symbol e})"
plot "fpr17vqf" using 1:2 ls 1 title "Vector Quotient Filter", \
       "fpr17cvqf" using 1:2 ls 2 title "Counting Vector Quotient Filter"
