set term png
set output "17remove.png"
set key out top center horizontal
set style data linespoints
set style line 1 lc rgb "dark-red" pt 7 ps 0.7 lt 1 lw 2
set style line 2 lc rgb "dark-green" pt 7 ps 0.7 lt 1 lw 2
set style line 3 lc rgb "dark-violet" pt 7 ps 0.7 lt 1 lw 2
set style line 4 lc rgb "blue" pt 7 ps 0.7 lt 1 lw 2
set style line 5 lc rgb "coral" pt 7 ps 0.7 lt 1 lw 2
set yrange [0:]
set grid ytics
set xlabel "Load Factor (%)" font ",16"
set ylabel "Throughput (Million ops. / second)" font ",16"
plot "17removevqf" using 1:2 ls 1 title "Vector Quotient Filter", \
       "17removecvqf" using 1:2 ls 2 title "Counting Vector Quotient Filter", \
       "17removecuckoo" using 1:2 ls 3 title "Cuckoo Filter", \
       "17removemorton" using 1:2 ls 4 title "Morton Filter", \
       "17removeqf" using 1:2 ls 5 title "Quotient Filter"
