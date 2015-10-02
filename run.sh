#!/bin/bash

binaries2="
  matrix-gcc-4.7-O3-unroll \
           "
#binaries2="matrix-clang matrix-clang-O3 matrix-clang-O3-unroll\
#  matrix-gcc-4.4-O0 matrix-gcc-4.4-O3 matrix-gcc-4.4-O3-unroll \
#  matrix-gcc-4.7-O0 matrix-gcc-4.7-O3 matrix-gcc-4.7-O3-unroll"

for bin in $binaries2;
do
  if [ -f $bin ]; then
    binaries="$binaries $bin"
  fi
done;


sizes="128 256 512 1024 2048 "
#512 1024 2048"

funcs="blas vanilla vanilla_omp vanilla2 vanilla2_omp avx avx_omp"
funcs+=" unroll2a unroll4a unroll2b unroll4b"

funcs2="stride block block_sse block_sse_omp block_avx block_avx_omp "
blocksize="32"
#512"

function plot
{
  if [ $# -ne 2 ]; then
    echo "plot name index" 2>&1 
    exit
  fi

  echo "plot $@"

  command="set terminal pdfcairo;
  set output '$1.pdf';
  set xlabel 'Size';
  set ylabel '$1';
  set xtics rotate '-45';
  set xrange [0:];
  set key font ',6'
  "

  plot="plot"
  for bin in $binaries; do
    for fun in $funcs; do
      command="$command $plot '${bin}-${fun}.dat' using 1:$2:xticlabels(1) with linespoints title '${bin}-${fun}'"
      plot=","
    done;

    for fun in $funcs2; do
      for bs in $blocksize; do
        command="$command $plot '${bin}-${fun}-${bs}.dat' using 1:$2:xticlabels(1) with linespoints title '${bin}-${fun}-${bs}'"
        plot=","
      done;
    done;

  done
  
  echo "$command" | gnuplot
}

function plot2
{
  if [ $# -ne 3 ]; then
    echo "plot name index" 2>&1 
    exit
  fi

  echo "plot $@"

  command="set terminal pdfcairo;
  set title 'blocks $1';
  set output '$1-$2.pdf';
  set xlabel 'Size';
  set ylabel '$2';
  set xtics rotate '-45';
  set xrange [0:];
  set key font ',6'
  "

  plot="plot"

  for fun in $funcs2; do
    for bs in $blocksize; do
      command="$command $plot '$1-${fun}-${bs}.dat' using 1:$3:xticlabels(1) with linespoints title '$1-${fun}-${bs}'"
      plot=","
    done;
  done;

  
  echo "$command" | gnuplot
}

function run
{
  echo "run"
  for bin in $binaries; do
    for fun in $funcs; do
      name=${bin}-${fun}
      echo -n "" > "${name}.dat"

      for size in $sizes; do
        echo -n "./$bin $size $size $size ${fun} " 1>&2
        echo -n "$size " >> "${name}.dat"
        ./$bin $size $size $size ${fun} >> "${name}.dat"
      done;
    done;

    for fun in $funcs2; do
      for bs in $blocksize; do
        name="${bin}-${fun}-${bs}"
        echo -n "" > "${name}.dat"

        for size in $sizes; do
          if [ $size -lt $bs ]; then 
            continue
          fi

          echo -n "./$bin $size $size $size ${fun} ${bs} " 1>&2
          echo -n "$size " >> "${name}.dat"
          ./$bin $size $size $size ${fun} ${bs} >> "${name}.dat"
        done;
      done;
    done;

  done;
}

function plotall
{
  plot "time" 2
  plot "gup" 3
  plot "gb" 4
  for bin in $binaries; do
    plot2 "$bin" "time" 2
    plot2 "$bin" "gup" 3
    plot2 "$bin" "gb" 4
  done;
}

echo "Binaries:$binaries"

if [ $# -ne 0 ] ; then
  $@
else
  run
  plotall
fi
# vim: set ts=2 sw=2 tw=0 ft=sh nolinebreak et :
