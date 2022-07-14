set term eps
set lmargin 16
set rmargin 15
set bmargin 3
set offset -0.8,-0.7,0,0
set output "memfpr2120.eps"
set key out top center horizontal font ",30"
set style data histograms
set style histogram cluster
set style fill solid
set yrange [0:]
set ytic 400 font ",30"
set y2tic 0.4 font ",30"
set ytics nomirror
set y2range [0:1.5]
set y2tics
set xtics scale 0
set ylabel "memory (kB)" offset -5 font ",30"
set y2label "fpr (%)" offset 4 font ",30"
set xtic offset 0,-0.5 font ",24`"
plot "memfpr2120" using 2 ti "memory" axis x1y1, \
	"memfpr2120" using 3:xticlabels(1) ti "fpr" axis x1y2
