set term eps
set output "memfpr1790.eps"
set key out top center horizontal
set style data histograms
set style histogram cluster
set style fill solid
set yrange [0:]
set ytics nomirror
set y2range [0:1.5]
set y2tics
set ylabel "memory (kB)" font ",16"
set y2label "fpr (%)" font ",16"
plot "memfpr1790" using 2 ti "memory" axis x1y1, \
	"memfpr1790" using 3:xticlabels(1) ti "fpr" axis x1y2
