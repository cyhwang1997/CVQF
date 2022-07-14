set term eps
set lmargin 17
set bmargin 5
set output "fpr.eps"
set key out top center horizontal font ",30"
set style data linespoints
set style line 1 lc rgb "dark-red" pt 7 ps 0.7 lt 1 lw 2
set style line 2 lc rgb "dark-green" pt 7 ps 0.7 lt 1 lw 2
set logscale xy 2
set grid ytics
set ytic font ",22"
set xtic font ",30"
set xlabel "log_2(100{/Symbol a})" offset 0,-1 font ",30"
set ylabel "-log_2({/Symbol e})" offset -6 font ",30"
plot "fprvqf" using 1:2 ls 1 title "VQF", \
       "fprcvqf" using 1:2 ls 2 title "CVQF"
